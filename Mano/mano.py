# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu (original release)
# Refactor: self-contained HF dataset generator + validation utilities

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hf_dataset_utils import labels_from_loss_mask, make_hf_dataset


def _ops_from_ttype(ttype: str) -> List[str]:
    ops: List[str] = []
    if "a" in ttype:
        ops.append("+")
    if "s" in ttype:
        ops.append("-")
    if "m" in ttype:
        ops.append("*")
    if "d" in ttype:
        ops.append("/")
    return ops


def _encode_op(op: str, rng: random.Random, knowledge_augment: bool) -> int:
    if op == "+":
        return rng.choice([5, 1] if knowledge_augment else [1])
    if op == "-":
        return rng.choice([6, 2] if knowledge_augment else [2])
    if op == "*":
        return rng.choice([7, 3] if knowledge_augment else [3])
    if op == "/":
        return rng.choice([8, 4] if knowledge_augment else [4])
    raise ValueError(f"unsupported op {op}")


def _eval_op(op: str, x: int, y: int, value_mod: int) -> int | None:
    if op == "+":
        return (x + y) % value_mod
    if op == "-":
        return (x - y + value_mod) % value_mod
    if op == "*":
        return (x * y) % value_mod
    if op == "/":
        if y == 0:
            return None
        return (x * pow(y, -1, value_mod)) % value_mod
    raise ValueError(f"unsupported op {op}")


def generate_mano_example(rng: random.Random, config: Mapping[str, Any]) -> Dict[str, object]:
    cfg = dict(config)

    qids = [a for a in range(1, cfg["L"] + 1)]
    lens = [a for a in range(1, cfg["L"] + 1)]
    this_idx = rng.randint(0, len(lens) - 1)
    this_len = lens[this_idx]
    this_qid = qids[this_idx]
    ops = _ops_from_ttype(cfg["ttype"])

    def gen(ll: int) -> Tuple[List[int], int | None]:
        if ll == 0:
            vid = rng.randint(0, cfg["value_mod"] - 1)
            return [5000 + vid], vid

        op = rng.choice(ops)
        l1 = rng.randint(0, ll - 1)
        seq1, val1 = gen(l1)
        seq2, val2 = gen(ll - 1 - l1)

        opid = _encode_op(op, rng, cfg["knowledge_augment"])
        seq = [opid] + seq1 + seq2
        if val1 is None or val2 is None:
            return seq, None
        return seq, _eval_op(op, val1, val2, cfg["value_mod"])

    while True:
        question, ans = gen(this_len)
        if ans is not None:
            break

    input_ids = [
        cfg["bos_token_id"] - (rng.randint(0, 1) if cfg["knowledge_augment"] else 0)
    ]
    input_ids += [this_qid * 10 + (rng.randint(0, 1) if cfg["knowledge_augment"] else 0)]
    input_ids += question
    input_ids += [this_qid * 10 + 4 + (rng.randint(0, 1) if cfg["knowledge_augment"] else 0)]
    input_ids += [5000 + ans]

    loss_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "labels": labels_from_loss_mask(input_ids, loss_mask),
        "length": this_len,
        "qid": this_qid,
    }


def _parse_prefix_expr(tokens: Sequence[int], value_mod: int) -> Tuple[int, int]:
    op_map = {
        1: "+",
        2: "-",
        3: "*",
        4: "/",
        5: "+",
        6: "-",
        7: "*",
        8: "/",
    }

    def rec(i: int) -> Tuple[int, int]:
        tok = tokens[i]
        if tok >= 5000:
            return tok - 5000, i + 1
        if tok not in op_map:
            raise ValueError("invalid expression token")
        lhs, i = rec(i + 1)
        rhs, i = rec(i)
        out = _eval_op(op_map[tok], lhs, rhs, value_mod)
        if out is None:
            raise ValueError("division by zero in expression")
        return out, i

    return rec(0)


def validate_mano_example(input_ids: Sequence[int], config: Mapping[str, Any]) -> bool:
    cfg = dict(config)
    if len(input_ids) < 5:
        return False
    if input_ids[-1] < 5000:
        return False

    expr_tokens = input_ids[2:-2]
    if not expr_tokens:
        return False

    try:
        expected, consumed = _parse_prefix_expr(expr_tokens, cfg["value_mod"])
    except ValueError:
        return False
    if consumed != len(expr_tokens):
        return False

    return input_ids[-1] == 5000 + expected


def make_mano_hf_dataset(num_examples: int, config: Mapping[str, Any], seed: int = 42):
    cfg = dict(config)
    return make_hf_dataset(
        num_examples=num_examples,
        seed=seed,
        sample_fn=lambda rng: generate_mano_example(rng, cfg),
    )
