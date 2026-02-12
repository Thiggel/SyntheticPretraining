from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.olmo3 import Olmo3Config, Olmo3ForCausalLM, Olmo3Model

from .recurrent_mixins import ActMixin, LoopStateMixin
from .recurrent_state_attention import RecurrentStateAttentionRouter


@dataclass
class RecurrentBaseModelOutput(BaseModelOutputWithPast):
    step_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    exit_probs: torch.FloatTensor | None = None


@dataclass
class RecurrentCausalLMOutput(CausalLMOutputWithPast):
    act_exit_probs: torch.FloatTensor | None = None
    step_losses: torch.FloatTensor | None = None
    act_kl_loss: torch.FloatTensor | None = None


class RecurrentOlmo3Model(LoopStateMixin, ActMixin, Olmo3Model):
    def __init__(self, config: Olmo3Config, recurrent_cfg: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.recurrent_cfg = recurrent_cfg

        self.num_loops = recurrent_cfg["num_loops"]
        self.encoder_layers = recurrent_cfg["encoder_layers"]
        self.loop_layers = recurrent_cfg["loop_layers"]
        self.decoder_layers = recurrent_cfg["decoder_layers"]

        self.inject_input_each_step = recurrent_cfg["inject_input_each_step"]
        self.random_init_loop_state = recurrent_cfg["random_init_loop_state"]
        self.random_init_std = recurrent_cfg["random_init_std"]
        self.tbptt_steps = recurrent_cfg["tbptt_steps"]
        self.train_recursion_mode = recurrent_cfg["train_recursion_mode"]
        self.poisson_mode_offset = recurrent_cfg["poisson_mode_offset"]
        self.loop_attention_mode = recurrent_cfg["loop_attention_mode"]

        act_cfg = recurrent_cfg["act"]
        self.act_enabled = act_cfg["enabled"]
        self.act_kl_weight = act_cfg["kl_weight"]

        self.init_inject(config.hidden_size)
        self.init_act(config.hidden_size)
        self.state_attention_router = RecurrentStateAttentionRouter(
            layer_types=self.config.layer_types,
            layer_offset=self.encoder_layers,
            mode=self.loop_attention_mode,
        )

    def _layer_groups(self) -> tuple[list[nn.Module], list[nn.Module], list[nn.Module]]:
        e0 = self.encoder_layers
        e1 = self.encoder_layers + self.loop_layers
        e2 = e1 + self.decoder_layers
        return list(self.layers[:e0]), list(self.layers[e0:e1]), list(self.layers[e1:e2])

    def _run_layer_group(
        self,
        hidden_states: torch.Tensor,
        layers: list[nn.Module],
        *,
        layer_offset: int,
        causal_mask_mapping: dict[str, torch.Tensor | None],
        position_ids: torch.LongTensor,
        cache_position: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **flash_attn_kwargs: Any,
    ) -> torch.Tensor:
        for local_idx, layer in enumerate(layers):
            cfg_attention_type = self.config.layer_types[layer_offset + local_idx]
            attention_type = getattr(layer.self_attn, "attention_type", cfg_attention_type)
            layer_mask = causal_mask_mapping[attention_type]
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    partial(layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    layer_mask,
                    position_ids,
                    None,
                    False,
                    cache_position,
                    position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **flash_attn_kwargs: Any,
    ) -> RecurrentBaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        num_loops = self.num_loops_for_forward(training=self.training)

        encoder_group, loop_group, decoder_group = self._layer_groups()

        hidden_states = self._run_layer_group(
            hidden_states,
            encoder_group,
            layer_offset=0,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        source_states = hidden_states
        loop_state = self.init_loop_state(source_states)

        step_hidden_states: list[torch.Tensor] = []
        step_exit_scores: list[torch.Tensor] = []

        for step_idx in range(max(num_loops, 1)):
            if num_loops > 0:
                loop_state = self.maybe_inject_source(source_states, loop_state)
                loop_state = self.state_attention_router.run(
                    hidden_states=loop_state,
                    source_states=source_states,
                    layers=loop_group,
                    causal_mask_mapping=causal_mask_mapping,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    training=self.training,
                    run_self_group=self._run_layer_group,
                    **flash_attn_kwargs,
                )
            else:
                loop_state = source_states

            decoded_states = self._run_layer_group(
                loop_state,
                decoder_group,
                layer_offset=self.encoder_layers + self.loop_layers,
                causal_mask_mapping=causal_mask_mapping,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            decoded_states = self.norm(decoded_states)

            if output_hidden_states:
                all_hidden_states += (decoded_states,)

            step_hidden_states.append(decoded_states)
            if self.act_enabled:
                step_exit_scores.append(self.step_exit_score(decoded_states))

            if num_loops > 0:
                loop_state = self.maybe_truncate_grad(loop_state, step_idx=step_idx, total_steps=num_loops)

        exit_probs = self.normalize_exit_scores(step_exit_scores) if self.act_enabled else None

        return RecurrentBaseModelOutput(
            last_hidden_state=step_hidden_states[-1],
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            step_hidden_states=tuple(step_hidden_states),
            exit_probs=exit_probs,
        )


class RecurrentOlmo3ForCausalLM(ActMixin, Olmo3ForCausalLM):
    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        self.model_cfg = model_cfg
        super().__init__(Olmo3Config(**model_cfg["hf"]))
        self.model = RecurrentOlmo3Model(self.config, model_cfg["recurrent"])
        self.recurrent_cfg = model_cfg["recurrent"]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> RecurrentCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])

        loss = None
        act_exit_probs = outputs.exit_probs
        step_losses = None
        act_kl_loss = None

        if labels is not None:
            if self.model.act_enabled:
                loss, step_losses, act_kl_loss = self.compute_act_loss(
                    step_hidden_states=outputs.step_hidden_states,
                    exit_probs=act_exit_probs,
                    labels=labels,
                    slice_indices=slice_indices,
                    lm_head=self.lm_head,
                    loss_function=self.loss_function,
                    vocab_size=self.config.vocab_size,
                    act_kl_weight=self.model.act_kl_weight,
                    loss_kwargs=kwargs,
                )
            else:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return RecurrentCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            act_exit_probs=act_exit_probs,
            step_losses=step_losses,
            act_kl_loss=act_kl_loss,
        )
