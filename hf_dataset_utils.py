"""Shared helpers for synthetic task generators.

This module keeps dataset plumbing centralized so each task file can focus on
its own generation and validation logic.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List

try:
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover - fallback for lean environments
    class Dataset(list):  # type: ignore[misc]
        """Small fallback with the subset of Hugging Face Dataset API we use."""

        @classmethod
        def from_list(cls, rows):
            return cls(rows)


def labels_from_loss_mask(
    input_ids: List[int],
    loss_mask: List[int],
    *,
    ignore_index: int = -100,
) -> List[int]:
    """Convert a 0/1 mask into autoregressive labels aligned to input_ids."""
    if len(input_ids) != len(loss_mask):
        raise ValueError("input_ids and loss_mask must have the same length")
    return [tok if mask else ignore_index for tok, mask in zip(input_ids, loss_mask)]


def make_hf_dataset(
    *,
    num_examples: int,
    seed: int,
    sample_fn: Callable[[random.Random], Dict[str, object]],
) -> Dataset:
    """Build a Hugging Face Dataset by repeatedly calling sample_fn."""
    rng = random.Random(seed)
    rows = [sample_fn(rng) for _ in range(num_examples)]
    return Dataset.from_list(rows)


def sample_n_with_inverse_sqrt_bias(rng: random.Random, n_max: int) -> int:
    """Sample n in [3, n_max] with weights proportional to 1 / (sqrt(N) + n).

    This matches the published generator behavior for Depo/Brevo.
    """
    if n_max < 3:
        raise ValueError("n_max must be >= 3")
    choices = list(range(3, n_max + 1))
    bias = n_max**0.5
    weights = [1.0 / (i + bias + 1e-12) for i in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def format_example_for_cli(example: Dict[str, object], max_tokens: int = 120) -> str:
    """Compact CLI rendering for quick manual inspection."""
    input_ids = example.get("input_ids", [])
    loss_mask = example.get("loss_mask", [])
    clipped_ids = input_ids[:max_tokens]
    clipped_mask = loss_mask[:max_tokens]
    suffix = "..." if len(input_ids) > max_tokens else ""
    return (
        f"len={len(input_ids)}\n"
        f"input_ids={clipped_ids}{suffix}\n"
        f"loss_mask={clipped_mask}{suffix}"
    )
