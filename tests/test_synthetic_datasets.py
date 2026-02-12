from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import BrevoTask, CapoTask, DepoTask, LanoTask, ManoTask


def test_depo_generation_and_validation() -> None:
    task = DepoTask(
        {
            "N": 225,
            "K": 8,
            "M": 10,
            "qa": False,
            "separator": False,
            "mini_vocab": 50,
            "min_tlen": 1,
            "max_tlen": 2,
            "fixed_k": None,
            "bos_token_id": 0,
            "separator_token_id": 9700,
            "query_token_base": 9000,
            "answer_token_id": 9500,
        }
    )
    rng = random.Random(0)
    for _ in range(60):
        ex = task.generate_example(rng)
        assert len(ex["input_ids"]) == len(ex["loss_mask"]) == len(ex["labels"])
        assert any(ex["loss_mask"])
        assert task.validate_example(ex)


def test_depo_token_override_and_fixed_k() -> None:
    task = DepoTask(
        {
            "N": 125,
            "K": 16,
            "M": 10,
            "qa": False,
            "separator": True,
            "mini_vocab": 3,
            "min_tlen": 5,
            "max_tlen": 7,
            "fixed_k": 4,
            "bos_token_id": 11,
            "separator_token_id": 12,
            "query_token_base": 300,
            "answer_token_id": 500,
        }
    )
    ex = task.generate_example(random.Random(7))
    assert ex["input_ids"][0] == 11
    assert 12 in ex["input_ids"]
    assert task.validate_example(ex)


def test_brevo_generation_and_validation() -> None:
    tasks = [
        BrevoTask(
            {
                "N": 70,
                "multi": False,
                "qa": False,
                "max_in": 4,
                "bos_token_id": 9999,
                "eos_token_id": 9998,
                "query_token_id": 9997,
                "answer_token_id": 9998,
            }
        ),
        BrevoTask(
            {
                "N": 30,
                "multi": True,
                "qa": False,
                "max_in": 4,
                "bos_token_id": 9999,
                "eos_token_id": 9998,
                "query_token_id": 9997,
                "answer_token_id": 9998,
            }
        ),
    ]
    for task in tasks:
        rng = random.Random(1)
        for _ in range(30):
            ex = task.generate_example(rng)
            assert len(ex["input_ids"]) == len(ex["token_type"]) == len(ex["loss_mask"]) == len(
                ex["labels"]
            )
            assert task.validate_example(ex)


def test_brevo_token_override() -> None:
    task = BrevoTask(
        {
            "N": 40,
            "multi": True,
            "qa": False,
            "max_in": 4,
            "bos_token_id": 101,
            "eos_token_id": 102,
            "query_token_id": 103,
            "answer_token_id": 102,
        }
    )
    ex = task.generate_example(random.Random(8))
    assert ex["input_ids"][0] == 101
    assert 103 in ex["input_ids"]
    assert task.validate_example(ex)


def test_mano_generation_and_validation() -> None:
    task = ManoTask(
        {
            "L": 10,
            "ttype": "asm",
            "value_mod": 23,
            "knowledge_augment": True,
            "bos_token_id": 9999,
        }
    )
    rng = random.Random(2)
    for _ in range(80):
        ex = task.generate_example(rng)
        assert len(ex["input_ids"]) == len(ex["loss_mask"]) == len(ex["labels"])
        assert task.validate_example(ex)


def test_mano_division_mode_stays_valid() -> None:
    task = ManoTask(
        {
            "L": 8,
            "ttype": "asmd",
            "value_mod": 23,
            "knowledge_augment": False,
            "bos_token_id": 9999,
        }
    )
    rng = random.Random(3)
    for _ in range(40):
        ex = task.generate_example(rng)
        assert task.validate_example(ex)


def test_mano_token_override() -> None:
    task = ManoTask(
        {
            "L": 6,
            "ttype": "asm",
            "value_mod": 23,
            "knowledge_augment": False,
            "bos_token_id": 77,
        }
    )
    ex = task.generate_example(random.Random(9))
    assert ex["input_ids"][0] == 77
    assert task.validate_example(ex)


def test_capo_generation_and_validation() -> None:
    task = CapoTask(
        {
            "num_people": 200,
            "exposures": 5,
            "reverse_md": False,
            "order": 0,
        }
    )
    rows = task.take(seed=4, count=20)
    for row in rows:
        assert isinstance(row["text"], str)
        assert len(row["text"]) > 20
        assert task.validate_example(row)


def test_lano_generation_and_validation() -> None:
    task = LanoTask({"config_name": "cfg3f"})
    rng = random.Random(5)
    for _ in range(3):
        ex = task.generate_example(rng)
        assert len(ex["input_ids"]) == len(ex["loss_mask"]) == len(ex["labels"])
        assert task.validate_example(ex)

    corrupted = list(ex["input_ids"])
    corrupted[0] = 99
    assert not task.validate_input_ids(corrupted)

