from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb, eager_attention_forward


class RecurrentStateAttentionRouter:
    def __init__(self, *, layer_types: list[str], layer_offset: int, mode: str) -> None:
        self.layer_types = layer_types
        self.layer_offset = layer_offset
        self.mode = mode

    def _layer_mask(
        self,
        *,
        layer: nn.Module,
        local_idx: int,
        causal_mask_mapping: dict[str, torch.Tensor | None],
    ) -> torch.Tensor | None:
        cfg_attention_type = self.layer_types[self.layer_offset + local_idx]
        attention_type = getattr(layer.self_attn, "attention_type", cfg_attention_type)
        return causal_mask_mapping[attention_type]

    def _cross_attention(
        self,
        *,
        layer: nn.Module,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        training: bool,
        **flash_attn_kwargs: Any,
    ) -> torch.Tensor:
        attn = layer.self_attn
        query_shape = (*query_states.shape[:-1], -1, attn.head_dim)
        kv_shape = (*kv_states.shape[:-1], -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(query_states)).view(query_shape).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(kv_states)).view(kv_shape).transpose(1, 2)
        v = attn.v_proj(kv_states).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, _ = attention_interface(
            attn,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not training else attn.attention_dropout,
            scaling=attn.scaling,
            **flash_attn_kwargs,
        )

        attn_output = attn_output.reshape(*query_states.shape[:-1], -1).contiguous()
        return attn.o_proj(attn_output)

    def _run_cross_group(
        self,
        *,
        hidden_states: torch.Tensor,
        source_states: torch.Tensor,
        layers: list[nn.Module],
        causal_mask_mapping: dict[str, torch.Tensor | None],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        training: bool,
        **flash_attn_kwargs: Any,
    ) -> torch.Tensor:
        for local_idx, layer in enumerate(layers):
            layer_mask = self._layer_mask(
                layer=layer,
                local_idx=local_idx,
                causal_mask_mapping=causal_mask_mapping,
            )

            residual = hidden_states
            hidden_states = self._cross_attention(
                layer=layer,
                query_states=hidden_states,
                kv_states=source_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                training=training,
                **flash_attn_kwargs,
            )
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.mlp(hidden_states)
            hidden_states = layer.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states

    def run(
        self,
        *,
        hidden_states: torch.Tensor,
        source_states: torch.Tensor,
        layers: list[nn.Module],
        causal_mask_mapping: dict[str, torch.Tensor | None],
        position_ids: torch.LongTensor,
        cache_position: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        training: bool,
        run_self_group: Callable[..., torch.Tensor],
        **flash_attn_kwargs: Any,
    ) -> torch.Tensor:
        if self.mode == "cross":
            return self._run_cross_group(
                hidden_states=hidden_states,
                source_states=source_states,
                layers=layers,
                causal_mask_mapping=causal_mask_mapping,
                position_embeddings=position_embeddings,
                training=training,
                **flash_attn_kwargs,
            )
        return run_self_group(
            hidden_states,
            layers,
            layer_offset=self.layer_offset,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
