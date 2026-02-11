"""Inspect any synthetic dataset via Hydra config.

Examples:
  python3 examples/inspect_dataset.py
  python3 examples/inspect_dataset.py dataset=brevo show=2
  python3 examples/inspect_dataset.py dataset=lano dataset.config.config_name=cfg3j
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hf_dataset_utils import format_example_for_cli


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_lano_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    if "config_path" not in cfg:
        cfg_name = cfg["config_name"]
        cfg_path = ROOT / "Lano-cfg" / "configs" / f"{cfg_name}.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"unknown CFG config: {cfg_path}")
        cfg["config_path"] = cfg_path
    return cfg


def _get_dataset_bundle(
    dataset_name: str,
    config: Dict[str, Any],
    *,
    num_examples: int,
    seed: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if dataset_name == "depo":
        from Depo.depo import make_depo_hf_dataset, validate_depo_example

        ds = make_depo_hf_dataset(num_examples=num_examples, config=config, seed=seed)
        return ds, validate_depo_example, config

    if dataset_name == "brevo":
        from Brevo.brevo import make_brevo_hf_dataset, validate_brevo_example

        ds = make_brevo_hf_dataset(num_examples=num_examples, config=config, seed=seed)
        return ds, validate_brevo_example, config

    if dataset_name == "mano":
        from Mano.mano import make_mano_hf_dataset, validate_mano_example

        ds = make_mano_hf_dataset(num_examples=num_examples, config=config, seed=seed)
        return ds, validate_mano_example, config

    if dataset_name == "lano":
        lano = _load_module(ROOT / "Lano-cfg" / "lano.py", "inspect_lano_mod")
        cfg = _resolve_lano_config(config)
        ds = lano.make_lano_hf_dataset(num_examples=num_examples, config=cfg, seed=seed)
        return ds, lano.validate_lano_example, cfg

    if dataset_name == "capo":
        capo = _load_module(ROOT / "Capo-bioS-bioR" / "Capo-bioS-bioR.py", "inspect_capo_mod")
        cfg = dict(config)
        ds = capo.make_capo_hf_dataset(
            num_examples=num_examples,
            config=cfg,
            seed=seed,
            base_dir=ROOT / "Capo-bioS-bioR",
        )
        return ds, capo.validate_capo_example, cfg

    raise ValueError(f"unsupported dataset: {dataset_name}")


@hydra.main(version_base=None, config_path="../conf/inspect_dataset", config_name="config")
def main(cfg: DictConfig) -> None:
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_name = str(ds_cfg["name"])
    dataset_config = dict(ds_cfg["config"])

    ds, validator, resolved_config = _get_dataset_bundle(
        dataset_name,
        dataset_config,
        num_examples=int(cfg.show),
        seed=int(cfg.seed),
    )

    print(f"dataset={dataset_name}")
    print(f"show={len(ds)}")

    for i in range(len(ds)):
        row = ds[i]
        if "input_ids" in row:
            is_valid = bool(validator(row["input_ids"], resolved_config))
            print(f"--- example {i} (valid={is_valid})")
            print(format_example_for_cli(row, max_tokens=int(cfg.max_tokens)))
        else:
            is_valid = bool(validator(row))
            text = str(row["text"])
            print(f"--- example {i} (valid={is_valid})")
            print(f"text_len={len(text)}")
            print(text[: int(cfg.text_chars)])


if __name__ == "__main__":
    main()
