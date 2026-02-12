from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import TaskBase, labels_from_loss_mask
from .lano_cfg import CFG_Config

ASSETS_DIR = Path(__file__).resolve().parent / "lano_assets"


class LanoTask(TaskBase):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = dict(config)
        if "config_path" not in cfg:
            cfg_name = str(cfg["config_name"])
            cfg["config_path"] = ASSETS_DIR / "configs" / f"{cfg_name}.json"
        super().__init__(cfg)

    def _cfg_graph(self):
        return CFG_Config.from_graph(str(self.config["config_path"]))

    def generate_example(self, rng: random.Random) -> dict[str, Any]:
        lcfg = self._cfg_graph()
        input_ids = lcfg.generate_onedata_pure(rng)
        loss_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "labels": labels_from_loss_mask(input_ids, loss_mask),
            "config_name": self.config["config_name"],
        }

    def validate_example(self, example: Mapping[str, Any]) -> bool:
        return self.validate_input_ids(example["input_ids"])

    def validate_input_ids(self, input_ids: Sequence[int]) -> bool:
        lcfg = self._cfg_graph()
        is_valid, _, _, _ = lcfg.solve_dp_noneq_fast(list(input_ids), no_debug=True)
        return is_valid == 0
