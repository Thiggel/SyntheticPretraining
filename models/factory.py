from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from .recurrent_olmo3 import RecurrentOlmo3ForCausalLM


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if tokenizer.bos_token is not None:
        tokenizer.pad_token = tokenizer.bos_token


def _build_from_pretrained(cfg: Mapping[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(cfg["name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(cfg["name_or_path"])
    _ensure_pad_token(tokenizer)
    return model, tokenizer


def _build_tiny_gpt2(cfg: Mapping[str, Any]):
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=int(cfg["vocab_size"]),
            n_positions=int(cfg["n_positions"]),
            n_ctx=int(cfg["n_ctx"]),
            n_embd=int(cfg["n_embd"]),
            n_layer=int(cfg["n_layer"]),
            n_head=int(cfg["n_head"]),
        )
    )
    return model, None


def _build_recurrent_olmo3(cfg: Mapping[str, Any]):
    model = RecurrentOlmo3ForCausalLM(cfg)

    tokenizer_name = cfg.get("tokenizer_name_or_path")
    if tokenizer_name is None and cfg.get("name_or_path"):
        tokenizer_name = cfg["name_or_path"]

    tokenizer = None
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        _ensure_pad_token(tokenizer)

    init_from = cfg.get("init_from_pretrained")
    if init_from:
        base_model = AutoModelForCausalLM.from_pretrained(init_from)
        model.load_state_dict(base_model.state_dict(), strict=False)

    return model, tokenizer


def build_model_and_tokenizer(cfg: Mapping[str, Any]):
    kind = str(cfg["kind"])
    if kind == "from_pretrained":
        return _build_from_pretrained(cfg)
    if kind == "tiny_gpt2":
        return _build_tiny_gpt2(cfg)
    if kind == "recurrent_olmo3":
        return _build_recurrent_olmo3(cfg)
    raise ValueError(f"unsupported model kind: {kind}")
