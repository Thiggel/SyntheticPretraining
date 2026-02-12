from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

try:
    from torch.utils.data import IterableDataset  # type: ignore
except Exception:  # pragma: no cover
    class IterableDataset:  # type: ignore[override]
        pass


Example = dict[str, Any]


class TaskBase(ABC):
    """Base class for synthetic tasks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @abstractmethod
    def generate_example(self, rng: random.Random) -> Example:
        """Sample one example."""

    @abstractmethod
    def validate_example(self, example: Mapping[str, Any]) -> bool:
        """Validate one sampled example."""

    def iter_examples(self, seed: int, num_examples: int | None = None) -> Iterator[Example]:
        rng = random.Random(seed)
        remaining = num_examples
        while remaining is None or remaining > 0:
            yield self.generate_example(rng)
            if remaining is not None:
                remaining -= 1

    def iterable(self, seed: int, num_examples: int | None = None) -> "TaskIterableDataset":
        return TaskIterableDataset(task=self, seed=seed, num_examples=num_examples)

    def take(self, seed: int, count: int) -> list[Example]:
        return list(self.iter_examples(seed=seed, num_examples=count))


class TaskIterableDataset(IterableDataset):
    """Torch-compatible iterable dataset that samples from a TaskBase."""

    def __init__(self, *, task: TaskBase, seed: int, num_examples: int | None = None) -> None:
        self.task = task
        self.seed = int(seed)
        self.num_examples = num_examples

    def __iter__(self):
        return self.task.iter_examples(seed=self.seed, num_examples=self.num_examples)


def labels_from_loss_mask(
    input_ids: Sequence[int],
    loss_mask: Sequence[int],
    *,
    ignore_index: int = -100,
) -> list[int]:
    if len(input_ids) != len(loss_mask):
        raise ValueError("input_ids and loss_mask must have same length")
    return [int(tok) if int(mask) else ignore_index for tok, mask in zip(input_ids, loss_mask)]


def sample_n_with_inverse_sqrt_bias(rng: random.Random, n_max: int) -> int:
    if n_max < 3:
        raise ValueError("n_max must be >= 3")
    choices = list(range(3, n_max + 1))
    bias = n_max**0.5
    weights = [1.0 / (i + bias + 1e-12) for i in choices]
    return int(rng.choices(choices, weights=weights, k=1)[0])
