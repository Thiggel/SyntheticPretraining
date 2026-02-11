# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu (original release in data_cfg.py)
# Refactor: lightweight wrapper for Hugging Face datasets + validation

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from hf_dataset_utils import labels_from_loss_mask, make_hf_dataset

_DATA_CFG_SPEC = importlib.util.spec_from_file_location(
    "lano_data_cfg", Path(__file__).resolve().parent / "data_cfg.py"
)
if _DATA_CFG_SPEC is None or _DATA_CFG_SPEC.loader is None:
    raise RuntimeError("cannot import data_cfg.py")
_DATA_CFG_MODULE = importlib.util.module_from_spec(_DATA_CFG_SPEC)
_DATA_CFG_SPEC.loader.exec_module(_DATA_CFG_MODULE)
CFG_Config = _DATA_CFG_MODULE.CFG_Config

def _load_cfg(config: Mapping[str, Any]) -> CFG_Config:
    return CFG_Config.from_graph(str(config["config_path"]))


def generate_lano_example(rng: random.Random, config: Mapping[str, Any]) -> Dict[str, object]:
    cfg = dict(config)
    lcfg = CFG_Config.from_graph(str(cfg["config_path"]))
    input_ids = lcfg.generate_onedata_pure(rng)
    loss_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "labels": labels_from_loss_mask(input_ids, loss_mask),
        "config_name": cfg["config_name"],
    }


def validate_lano_example(input_ids: Sequence[int], config: Mapping[str, Any]) -> bool:
    lcfg = _load_cfg(config)
    is_valid, _, _, _ = lcfg.solve_dp_noneq_fast(list(input_ids), no_debug=True)
    return is_valid == 0


def make_lano_hf_dataset(num_examples: int, config: Mapping[str, Any], seed: int = 42):
    cfg = dict(config)
    return make_hf_dataset(
        num_examples=num_examples,
        seed=seed,
        sample_fn=lambda rng: generate_lano_example(rng, cfg),
    )
