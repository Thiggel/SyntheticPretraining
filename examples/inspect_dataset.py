"""Inspect synthetic dataset samples via Hydra config."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import create_task

ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[92m"


def _maybe_load_tokenizer(name_or_path: str | None):
    if not name_or_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)


def _escape_token_text(text: str) -> str:
    t = text.replace("\n", "\\n").replace("\t", "\\t")
    return t if t else "<empty>"


def _synthetic_decode(dataset_name: str, token_id: int, config: dict[str, Any]) -> str:
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
    config: dict[str, Any],
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
    row: dict[str, Any],
    config: dict[str, Any],
    tokenizer,
    max_tokens: int,
    colorize_loss_mask: bool,
) -> str:
    input_ids = list(row["input_ids"])
    loss_mask = list(row["loss_mask"])
    if max_tokens <= 0:
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
            token_id=int(tok_id),
            config=config,
            tokenizer=tokenizer,
        )
        if colorize_loss_mask:
            color = ANSI_GREEN if m else ANSI_DIM
            decoded = f"{color}{decoded}{ANSI_RESET}"
        pieces.append(decoded)

    out = (
        f"len={len(input_ids)}\n"
        f"legend: dim=loss_mask0, green=loss_mask1\n"
        f"tokens={' '.join(pieces)}"
    )
    if clipped:
        out += f"\ntruncated=true shown={len(shown_ids)} total={len(input_ids)}"
    return out


@hydra.main(version_base=None, config_path="../conf/inspect_dataset", config_name="config")
def main(cfg: DictConfig) -> None:
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_name = str(ds_cfg["name"])
    dataset_config = dict(ds_cfg["config"])
    tokenizer = _maybe_load_tokenizer(cfg.tokenizer_name_or_path)

    task = create_task(dataset_name, dataset_config)
    rows = task.take(seed=int(cfg.seed), count=int(cfg.show))
    resolved_config = dict(task.config)

    print(f"dataset={dataset_name}")
    print(f"show={len(rows)}")

    for i, row in enumerate(rows):
        is_valid = bool(task.validate_example(row))
        print(f"--- example {i} (valid={is_valid})")
        if "input_ids" in row:
            print(
                _render_token_example(
                    dataset_name=dataset_name,
                    row=dict(row),
                    config=resolved_config,
                    tokenizer=tokenizer,
                    max_tokens=int(cfg.max_tokens),
                    colorize_loss_mask=bool(cfg.colorize_loss_mask),
                )
            )
        else:
            text = str(row["text"])
            print(f"text_len={len(text)}")
            print(text[: int(cfg.text_chars)])


if __name__ == "__main__":
    main()
