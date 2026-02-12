from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import TaskBase, TaskIterableDataset
from .brevo import BrevoTask
from .capo import CapoTask
from .depo import DepoTask
from .lano import LanoTask
from .mano import ManoTask

TASK_REGISTRY = {
    "depo": DepoTask,
    "brevo": BrevoTask,
    "mano": ManoTask,
    "lano": LanoTask,
    "capo": CapoTask,
}


def create_task(name: str, config: Mapping[str, Any]) -> TaskBase:
    key = str(name).lower()
    if key not in TASK_REGISTRY:
        raise ValueError(f"unsupported dataset: {name}")
    return TASK_REGISTRY[key](dict(config))


__all__ = [
    "TaskBase",
    "TaskIterableDataset",
    "DepoTask",
    "BrevoTask",
    "ManoTask",
    "LanoTask",
    "CapoTask",
    "create_task",
]
