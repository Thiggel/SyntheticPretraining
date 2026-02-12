from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import torch


class LoopStateMixin:
    inject_input_each_step: bool
    random_init_loop_state: bool
    random_init_std: float
    tbptt_steps: int
    inject_proj: torch.nn.Linear | None
    num_loops: int
    train_recursion_mode: str
    poisson_mode_offset: float

    def init_inject(self, hidden_size: int) -> None:
        self.inject_proj = None
        if self.inject_input_each_step:
            self.inject_proj = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def init_loop_state(self, source_states: torch.Tensor) -> torch.Tensor:
        if not self.random_init_loop_state:
            return source_states
        noise = torch.randn_like(source_states) * self.random_init_std
        return noise

    def maybe_inject_source(self, source_states: torch.Tensor, loop_state: torch.Tensor) -> torch.Tensor:
        if not self.inject_input_each_step:
            return loop_state
        merged = torch.cat([source_states, loop_state], dim=-1)
        return self.inject_proj(merged)

    def maybe_truncate_grad(self, loop_state: torch.Tensor, *, step_idx: int, total_steps: int) -> torch.Tensor:
        if self.tbptt_steps <= 0:
            return loop_state
        is_boundary = (step_idx + 1) % self.tbptt_steps == 0
        is_last = step_idx + 1 == total_steps
        if is_boundary and not is_last:
            return loop_state.detach()
        return loop_state

    def num_loops_for_forward(self, *, training: bool) -> int:
        if training and self.train_recursion_mode == "poisson":
            lam = torch.tensor(self.num_loops + self.poisson_mode_offset)
            return int(torch.poisson(lam).item())
        return self.num_loops


class ActMixin:
    act_enabled: bool
    act_head: torch.nn.Linear | None

    def init_act(self, hidden_size: int) -> None:
        self.act_head = None
        if self.act_enabled:
            self.act_head = torch.nn.Linear(hidden_size, 1)

    def step_exit_score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        exit_logits = self.act_head(hidden_states)
        return torch.sigmoid(exit_logits).mean()

    def normalize_exit_scores(self, scores: list[torch.Tensor]) -> torch.Tensor:
        if not scores:
            return torch.empty(0)
        stacked = torch.stack(scores)
        denom = stacked.sum() + 1e-8
        return stacked / denom

    def kl_to_uniform(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.numel() == 0:
            return probs.new_zeros(())
        uniform = torch.full_like(probs, 1.0 / float(probs.numel()))
        return torch.sum(probs * ((probs + 1e-8).log() - (uniform + 1e-8).log()))

    def compute_act_loss(
        self,
        *,
        step_hidden_states: Sequence[torch.Tensor],
        exit_probs: torch.Tensor,
        labels: torch.Tensor,
        slice_indices: slice | torch.Tensor,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        loss_function: Callable[..., torch.Tensor],
        vocab_size: int,
        act_kl_weight: float,
        loss_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step_logits = [lm_head(h[:, slice_indices, :]) for h in step_hidden_states]
        step_losses = torch.stack(
            [loss_function(logits=lg, labels=labels, vocab_size=vocab_size, **loss_kwargs) for lg in step_logits]
        )
        weighted_probs = exit_probs.to(step_losses.dtype)
        act_kl_loss = self.kl_to_uniform(weighted_probs)
        weighted_loss = torch.sum(weighted_probs * step_losses)
        total_loss = weighted_loss + act_kl_weight * act_kl_loss
        return total_loss, step_losses, act_kl_loss
