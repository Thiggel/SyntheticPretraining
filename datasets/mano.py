from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .base import TaskBase, labels_from_loss_mask


class ManoTask(TaskBase):
    @staticmethod
    def _ops_from_ttype(ttype: str) -> list[str]:
        ops: list[str] = []
        if "a" in ttype:
            ops.append("+")
        if "s" in ttype:
            ops.append("-")
        if "m" in ttype:
            ops.append("*")
        if "d" in ttype:
            ops.append("/")
        return ops

    @staticmethod
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

    @staticmethod
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

    def _generate_expression(
        self,
        rng: random.Random,
        length: int,
        ops: Sequence[str],
    ) -> tuple[list[int], int | None]:
        if length == 0:
            value = rng.randint(0, int(self.config["value_mod"]) - 1)
            return [5000 + value], value

        op = rng.choice(list(ops))
        left_len = rng.randint(0, length - 1)
        seq1, val1 = self._generate_expression(rng, left_len, ops)
        seq2, val2 = self._generate_expression(rng, length - 1 - left_len, ops)
        op_id = self._encode_op(op, rng, bool(self.config["knowledge_augment"]))
        seq = [op_id] + seq1 + seq2
        if val1 is None or val2 is None:
            return seq, None
        return seq, self._eval_op(op, val1, val2, int(self.config["value_mod"]))

    def generate_example(self, rng: random.Random) -> dict[str, Any]:
        L = int(self.config["L"])
        this_len = rng.randint(1, L)
        this_qid = this_len
        ops = self._ops_from_ttype(str(self.config["ttype"]))

        while True:
            question, answer = self._generate_expression(rng, this_len, ops)
            if answer is not None:
                break

        ka = bool(self.config["knowledge_augment"])
        bos = int(self.config["bos_token_id"]) - (rng.randint(0, 1) if ka else 0)
        qprefix = this_qid * 10 + (rng.randint(0, 1) if ka else 0)
        qsuffix = this_qid * 10 + 4 + (rng.randint(0, 1) if ka else 0)

        input_ids = [bos, qprefix] + question + [qsuffix, 5000 + int(answer)]
        loss_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "labels": labels_from_loss_mask(input_ids, loss_mask),
            "length": this_len,
            "qid": this_qid,
        }

    def validate_example(self, example: Mapping[str, Any]) -> bool:
        return self.validate_input_ids(example["input_ids"])

    def validate_input_ids(self, input_ids: Sequence[int]) -> bool:
        if len(input_ids) < 5 or int(input_ids[-1]) < 5000:
            return False
        expr_tokens = input_ids[2:-2]
        if not expr_tokens:
            return False
        try:
            expected, consumed = self._parse_prefix_expr(expr_tokens, int(self.config["value_mod"]))
        except ValueError:
            return False
        if consumed != len(expr_tokens):
            return False
        return int(input_ids[-1]) == 5000 + expected

    def _parse_prefix_expr(self, tokens: Sequence[int], value_mod: int) -> tuple[int, int]:
        op_map = {1: "+", 2: "-", 3: "*", 4: "/", 5: "+", 6: "-", 7: "*", 8: "/"}

        def rec(i: int) -> tuple[int, int]:
            tok = int(tokens[i])
            if tok >= 5000:
                return tok - 5000, i + 1
            if tok not in op_map:
                raise ValueError("invalid expression token")
            lhs, i = rec(i + 1)
            rhs, i = rec(i)
            out = self._eval_op(op_map[tok], lhs, rhs, value_mod)
            if out is None:
                raise ValueError("division by zero")
            return out, i

        return rec(0)
