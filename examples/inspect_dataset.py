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

ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[92m"


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


def _maybe_load_tokenizer(name_or_path: str | None):
    if not name_or_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)


def _escape_token_text(text: str) -> str:
    t = text.replace("\n", "\\n").replace("\t", "\\t")
    return t if t else "<empty>"


def _synthetic_decode(dataset_name: str, token_id: int, config: Dict[str, Any]) -> str:
    if dataset_name == "depo":
        if token_id == config["bos_token_id"]:
            return "<bos>"
        if token_id == config["separator_token_id"]:
            return "<sep>"
        if token_id == config["answer_token_id"]:
            return "<ans>"
        if config["query_token_base"] < token_id < config["answer_token_id"]:
            return f"<query_{token_id - config['query_token_base']}>"
        return str(token_id)

    if dataset_name == "brevo":
        if token_id == config["bos_token_id"]:
            return "<bos>"
        if token_id == config["eos_token_id"]:
            return "<eos>"
        if token_id == config["query_token_id"]:
            return "<query>"
        if token_id == config["answer_token_id"]:
            return "<ans>"
        return str(token_id)

    if dataset_name == "mano":
        op_map = {
            1: "<+>",
            2: "<->",
            3: "<*>",
            4: "</>",
            5: "<+a>",
            6: "<-a>",
            7: "<*a>",
            8: "</a>",
        }
        if token_id in op_map:
            return op_map[token_id]
        if token_id >= 5000:
            return f"<v{token_id - 5000}>"
        return str(token_id)

    return str(token_id)


def _decode_token(
    *,
    dataset_name: str,
    token_id: int,
    config: Dict[str, Any],
    tokenizer,
) -> str:
    if tokenizer is not None:
        try:
            tok = tokenizer.convert_ids_to_tokens(int(token_id))
            if tok is None:
                tok = tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            return _escape_token_text(str(tok))
        except Exception:
            pass
    return _synthetic_decode(dataset_name, int(token_id), config)


def _render_token_example(
    *,
    dataset_name: str,
    row: Dict[str, Any],
    config: Dict[str, Any],
    tokenizer,
    max_tokens: int,
    colorize_loss_mask: bool,
) -> str:
    input_ids = list(row["input_ids"])
    loss_mask = list(row["loss_mask"])
    if max_tokens is None or int(max_tokens) <= 0:
        shown_ids = input_ids
        shown_mask = loss_mask
        clipped = False
    else:
        shown_ids = input_ids[:max_tokens]
        shown_mask = loss_mask[:max_tokens]
        clipped = len(input_ids) > max_tokens

    pieces = []
    for tok_id, m in zip(shown_ids, shown_mask):
        decoded = _decode_token(
            dataset_name=dataset_name,
            token_id=tok_id,
            config=config,
            tokenizer=tokenizer,
        )
        part = decoded
        if colorize_loss_mask:
            color = ANSI_GREEN if m else ANSI_DIM
            part = f"{color}{part}{ANSI_RESET}"
        pieces.append(part)

    out = (
        f"len={len(input_ids)}\n"
        f"legend: dim=loss_mask0, green=loss_mask1\n"
        f"tokens={' '.join(pieces)}"
    )
    if clipped:
        out += f"\ntruncated=true shown={len(shown_ids)} total={len(input_ids)}"
    return out


def _get_dataset_bundle(
    dataset_name: str,
    config: Dict[str, Any],
    *,
    num_examples: int,
    seed: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if dataset_name == "depo":
        from Depo.depo import make_depo_hf_dataset, validate_depo_example

        cfg = dict(config)
        ds = make_depo_hf_dataset(num_examples=num_examples, config=cfg, seed=seed)
        return ds, validate_depo_example, cfg

    if dataset_name == "brevo":
        from Brevo.brevo import make_brevo_hf_dataset, validate_brevo_example

        cfg = dict(config)
        ds = make_brevo_hf_dataset(num_examples=num_examples, config=cfg, seed=seed)
        return ds, validate_brevo_example, cfg

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
    tokenizer = _maybe_load_tokenizer(cfg.tokenizer_name_or_path)

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
            print(
                _render_token_example(
                    dataset_name=dataset_name,
                    row=row,
                    config=resolved_config,
                    tokenizer=tokenizer,
                    max_tokens=int(cfg.max_tokens),
                    colorize_loss_mask=bool(cfg.colorize_loss_mask),
                )
            )
        else:
            is_valid = bool(validator(row))
            text = str(row["text"])
            print(f"--- example {i} (valid={is_valid})")
            print(f"text_len={len(text)}")
            print(text[: int(cfg.text_chars)])


if __name__ == "__main__":
    main()
