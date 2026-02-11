# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu (original release)
# Refactor: self-contained HF dataset generator + validation utilities

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hf_dataset_utils import (
    labels_from_loss_mask,
    make_hf_dataset,
    sample_n_with_inverse_sqrt_bias,
)


def generate_multi_token_words(
    rng: random.Random,
    n: int,
    mini_vocab: int = 3,
    min_tlen: int = 5,
    max_tlen: int = 7,
) -> List[List[int]]:
    def sample_word(length: int) -> Tuple[int, ...]:
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab
        return tuple(toks)

    words = set()
    while len(words) < n:
        words.add(sample_word(rng.randint(min_tlen, max_tlen)))
    return [list(word) for word in words]


def _qa_powers(k_max: int) -> List[int]:
    powers = [2**i for i in range(k_max.bit_length()) if 2**i <= k_max]
    if k_max == 32:
        powers.append(24)
    if powers[-1] != k_max:
        powers.append(k_max)
    return powers


def generate_depo_example(rng: random.Random, config: Mapping[str, Any]) -> Dict[str, object]:
    cfg = dict(config)

    n = cfg["N"] if cfg["qa"] else sample_n_with_inverse_sqrt_bias(rng, cfg["N"])

    vals = generate_multi_token_words(
        rng,
        n,
        mini_vocab=cfg["mini_vocab"],
        min_tlen=cfg["min_tlen"],
        max_tlen=cfg["max_tlen"],
    )
    rng.shuffle(vals)
    order = rng.sample(range(n), n)

    input_ids: List[int] = [cfg["bos_token_id"]]
    for i in range(n):
        if cfg["separator"]:
            input_ids.append(cfg["separator_token_id"])
        v1 = vals[order[i]]
        v2 = vals[(order[i] + 1) % n]
        input_ids.extend(v1 + v2)

    loss_mask: List[int] = [0] * len(input_ids)
    fixed_k = cfg["fixed_k"]
    powers = _qa_powers(cfg["K"]) if cfg["qa"] else None

    for idx in rng.sample(range(n), k=min(n, cfg["M"])):
        if fixed_k is not None:
            k = fixed_k
        else:
            k = rng.choice(powers) if powers is not None else rng.randint(1, cfg["K"])
        v1 = vals[idx]
        v2 = vals[(idx + k) % n]
        input_ids.extend([cfg["query_token_base"] + k] + v1 + [cfg["answer_token_id"]] + v2)
        loss_mask.extend([0] * (len(v1) + 1) + [1] * (len(v2) + 1))

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "labels": labels_from_loss_mask(input_ids, loss_mask),
        "n": n,
        "query_count": min(n, cfg["M"]),
        "K": cfg["K"],
        "mini_vocab": cfg["mini_vocab"],
    }


def _split_words(token_seq: Sequence[int], mini_vocab: int) -> List[Tuple[int, ...]]:
    words: List[Tuple[int, ...]] = []
    cur: List[int] = []
    for tok in token_seq:
        cur.append(tok)
        if mini_vocab < tok <= 2 * mini_vocab:
            words.append(tuple(cur))
            cur = []
    if cur:
        raise ValueError("incomplete trailing word")
    return words


def validate_depo_example(input_ids: Sequence[int], config: Mapping[str, Any]) -> bool:
    cfg = dict(config)
    if not input_ids or input_ids[0] != cfg["bos_token_id"]:
        return False

    first_query = None
    for i, tok in enumerate(input_ids[1:], start=1):
        if cfg["query_token_base"] < tok < cfg["answer_token_id"]:
            first_query = i
            break
    if first_query is None:
        return False

    edge_stream = [x for x in input_ids[1:first_query] if x != cfg["separator_token_id"]]
    try:
        edge_words = _split_words(edge_stream, cfg["mini_vocab"])
    except ValueError:
        return False
    if len(edge_words) % 2 != 0:
        return False

    successor: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    for i in range(0, len(edge_words), 2):
        src = edge_words[i]
        dst = edge_words[i + 1]
        if src in successor:
            return False
        successor[src] = dst

    idx = first_query
    while idx < len(input_ids):
        qtok = input_ids[idx]
        if not (cfg["query_token_base"] < qtok < cfg["answer_token_id"]):
            return False
        k = qtok - cfg["query_token_base"]
        idx += 1

        qword_tokens: List[int] = []
        while idx < len(input_ids) and input_ids[idx] != cfg["answer_token_id"]:
            if cfg["query_token_base"] < input_ids[idx] < cfg["answer_token_id"]:
                return False
            qword_tokens.append(input_ids[idx])
            idx += 1
        if idx >= len(input_ids):
            return False
        idx += 1

        aword_tokens: List[int] = []
        while idx < len(input_ids) and not (
            cfg["query_token_base"] < input_ids[idx] < cfg["answer_token_id"]
        ):
            aword_tokens.append(input_ids[idx])
            idx += 1

        try:
            qwords = _split_words(qword_tokens, cfg["mini_vocab"])
            awords = _split_words(aword_tokens, cfg["mini_vocab"])
        except ValueError:
            return False
        if len(qwords) != 1 or len(awords) != 1:
            return False

        cur = qwords[0]
        for _ in range(k):
            if cur not in successor:
                return False
            cur = successor[cur]
        if cur != awords[0]:
            return False

    return True


def make_depo_hf_dataset(num_examples: int, config: Mapping[str, Any], seed: int = 42):
    cfg = dict(config)
    return make_hf_dataset(
        num_examples=num_examples,
        seed=seed,
        sample_fn=lambda rng: generate_depo_example(rng, cfg),
    )
