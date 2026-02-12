from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


def generate_multi_token_words(
    rng: random.Random,
    n: int,
    *,
    mini_vocab: int,
    min_tlen: int,
    max_tlen: int,
) -> list[list[int]]:
    def sample_word(length: int) -> tuple[int, ...]:
        tokens = [rng.randint(1, mini_vocab) for _ in range(length)]
        tokens[-1] += mini_vocab
        return tuple(tokens)

    words = set()
    while len(words) < n:
        words.add(sample_word(rng.randint(min_tlen, max_tlen)))
    return [list(word) for word in words]


def split_multi_token_words(token_seq: Sequence[int], mini_vocab: int) -> list[tuple[int, ...]]:
    words: list[tuple[int, ...]] = []
    cur: list[int] = []
    for tok in token_seq:
        cur.append(int(tok))
        if mini_vocab < tok <= 2 * mini_vocab:
            words.append(tuple(cur))
            cur = []
    if cur:
        raise ValueError("incomplete trailing word")
    return words


def as_int_list(values: Iterable[int]) -> list[int]:
    return [int(x) for x in values]
