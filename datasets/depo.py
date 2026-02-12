from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .base import TaskBase, labels_from_loss_mask, sample_n_with_inverse_sqrt_bias
from .common import generate_multi_token_words, split_multi_token_words


class DepoTask(TaskBase):
    def _sample_n(self, rng: random.Random) -> int:
        if self.config["qa"]:
            return int(self.config["N"])
        return sample_n_with_inverse_sqrt_bias(rng, int(self.config["N"]))

    def _sample_hop(self, rng: random.Random) -> int:
        fixed_k = self.config["fixed_k"]
        if fixed_k is not None:
            return int(fixed_k)
        if self.config["qa"]:
            return int(rng.choice(self._qa_hops(int(self.config["K"]))))
        return int(rng.randint(1, int(self.config["K"])))

    @staticmethod
    def _qa_hops(k_max: int) -> list[int]:
        powers = [2**i for i in range(k_max.bit_length()) if 2**i <= k_max]
        if k_max == 32:
            powers.append(24)
        if powers[-1] != k_max:
            powers.append(k_max)
        return powers

    def _build_edge_tokens(self, rng: random.Random, n: int) -> tuple[list[list[int]], list[int]]:
        values = generate_multi_token_words(
            rng,
            n,
            mini_vocab=int(self.config["mini_vocab"]),
            min_tlen=int(self.config["min_tlen"]),
            max_tlen=int(self.config["max_tlen"]),
        )
        rng.shuffle(values)
        order = rng.sample(range(n), n)
        return values, order

    def _append_graph(self, input_ids: list[int], values: list[list[int]], order: list[int]) -> None:
        n = len(order)
        for i in range(n):
            if self.config["separator"]:
                input_ids.append(int(self.config["separator_token_id"]))
            src = values[order[i]]
            dst = values[(order[i] + 1) % n]
            input_ids.extend(src + dst)

    def _append_queries(
        self,
        rng: random.Random,
        input_ids: list[int],
        loss_mask: list[int],
        values: list[list[int]],
        n: int,
    ) -> None:
        query_count = min(n, int(self.config["M"]))
        for idx in rng.sample(range(n), k=query_count):
            k = self._sample_hop(rng)
            qword = values[idx]
            aword = values[(idx + k) % n]
            input_ids.extend(
                [int(self.config["query_token_base"]) + k] + qword + [int(self.config["answer_token_id"])] + aword
            )
            loss_mask.extend([0] * (len(qword) + 1) + [1] * (len(aword) + 1))

    def generate_example(self, rng: random.Random) -> dict[str, Any]:
        n = self._sample_n(rng)
        values, order = self._build_edge_tokens(rng, n)

        input_ids = [int(self.config["bos_token_id"])]
        self._append_graph(input_ids, values, order)

        loss_mask = [0] * len(input_ids)
        self._append_queries(rng, input_ids, loss_mask, values, n)

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "labels": labels_from_loss_mask(input_ids, loss_mask),
            "n": n,
            "query_count": min(n, int(self.config["M"])),
            "K": int(self.config["K"]),
            "mini_vocab": int(self.config["mini_vocab"]),
        }

    def validate_example(self, example: Mapping[str, Any]) -> bool:
        return self.validate_input_ids(example["input_ids"])

    def validate_input_ids(self, input_ids: Sequence[int]) -> bool:
        cfg = self.config
        if not input_ids or input_ids[0] != int(cfg["bos_token_id"]):
            return False

        first_query = self._first_query_index(input_ids)
        if first_query is None:
            return False

        successor = self._parse_successor_map(input_ids[1:first_query])
        if successor is None:
            return False

        return self._validate_queries(input_ids, first_query, successor)

    def _first_query_index(self, input_ids: Sequence[int]) -> int | None:
        qbase = int(self.config["query_token_base"])
        ans = int(self.config["answer_token_id"])
        for i, tok in enumerate(input_ids[1:], start=1):
            if qbase < int(tok) < ans:
                return i
        return None

    def _parse_successor_map(
        self, edge_stream_tokens: Sequence[int]
    ) -> dict[tuple[int, ...], tuple[int, ...]] | None:
        cfg = self.config
        edge_stream = [int(x) for x in edge_stream_tokens if int(x) != int(cfg["separator_token_id"])]
        try:
            edge_words = split_multi_token_words(edge_stream, int(cfg["mini_vocab"]))
        except ValueError:
            return None
        if len(edge_words) % 2 != 0:
            return None

        successor: dict[tuple[int, ...], tuple[int, ...]] = {}
        for i in range(0, len(edge_words), 2):
            src = edge_words[i]
            dst = edge_words[i + 1]
            if src in successor:
                return None
            successor[src] = dst
        return successor

    def _validate_queries(
        self,
        input_ids: Sequence[int],
        idx: int,
        successor: Mapping[tuple[int, ...], tuple[int, ...]],
    ) -> bool:
        cfg = self.config
        qbase = int(cfg["query_token_base"])
        ans_token = int(cfg["answer_token_id"])
        mini_vocab = int(cfg["mini_vocab"])

        while idx < len(input_ids):
            qtok = int(input_ids[idx])
            if not (qbase < qtok < ans_token):
                return False
            k = qtok - qbase
            idx += 1

            qword_tokens: list[int] = []
            while idx < len(input_ids) and int(input_ids[idx]) != ans_token:
                if qbase < int(input_ids[idx]) < ans_token:
                    return False
                qword_tokens.append(int(input_ids[idx]))
                idx += 1
            if idx >= len(input_ids):
                return False
            idx += 1

            aword_tokens: list[int] = []
            while idx < len(input_ids) and not (qbase < int(input_ids[idx]) < ans_token):
                aword_tokens.append(int(input_ids[idx]))
                idx += 1

            try:
                qwords = split_multi_token_words(qword_tokens, mini_vocab)
                awords = split_multi_token_words(aword_tokens, mini_vocab)
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
