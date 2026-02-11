from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Brevo.brevo import generate_brevo_example, validate_brevo_example
from Depo.depo import generate_depo_example, validate_depo_example
from Mano.mano import generate_mano_example, validate_mano_example


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


capo = _load_module("Capo-bioS-bioR/Capo-bioS-bioR.py", "capo_mod")
lano = _load_module("Lano-cfg/lano.py", "lano_mod")


def test_depo_generation_and_validation() -> None:
    cfg = {
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
    rng = random.Random(0)

    for _ in range(60):
        ex = generate_depo_example(rng, cfg)
        assert len(ex["input_ids"]) == len(ex["loss_mask"]) == len(ex["labels"])
        assert any(ex["loss_mask"])
        assert validate_depo_example(ex["input_ids"], cfg)


def test_depo_token_override_and_fixed_k() -> None:
    cfg = {
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
    ex = generate_depo_example(random.Random(7), cfg)
    assert ex["input_ids"][0] == 11
    assert 12 in ex["input_ids"]
    assert validate_depo_example(ex["input_ids"], cfg)


def test_brevo_generation_and_validation() -> None:
    cfgs = [
        {
            "N": 70,
            "multi": False,
            "qa": False,
            "max_in": 4,
            "bos_token_id": 9999,
            "eos_token_id": 9998,
            "query_token_id": 9997,
            "answer_token_id": 9998,
        },
        {
            "N": 30,
            "multi": True,
            "qa": False,
            "max_in": 4,
            "bos_token_id": 9999,
            "eos_token_id": 9998,
            "query_token_id": 9997,
            "answer_token_id": 9998,
        },
    ]
    for cfg in cfgs:
        rng = random.Random(1)
        for _ in range(30):
            ex = generate_brevo_example(rng, cfg)
            assert len(ex["input_ids"]) == len(ex["token_type"]) == len(ex["loss_mask"]) == len(
                ex["labels"]
            )
            assert validate_brevo_example(ex["input_ids"], cfg)


def test_brevo_token_override() -> None:
    cfg = {
        "N": 40,
        "multi": True,
        "qa": False,
        "max_in": 4,
        "bos_token_id": 101,
        "eos_token_id": 102,
        "query_token_id": 103,
        "answer_token_id": 102,
    }
    ex = generate_brevo_example(random.Random(8), cfg)
    assert ex["input_ids"][0] == 101
    assert 103 in ex["input_ids"]
    assert validate_brevo_example(ex["input_ids"], cfg)


def test_mano_generation_and_validation() -> None:
    cfg = {
        "L": 10,
        "ttype": "asm",
        "value_mod": 23,
        "knowledge_augment": True,
        "bos_token_id": 9999,
    }
    rng = random.Random(2)
    for _ in range(80):
        ex = generate_mano_example(rng, cfg)
        assert len(ex["input_ids"]) == len(ex["loss_mask"]) == len(ex["labels"])
        assert validate_mano_example(ex["input_ids"], cfg)


def test_mano_division_mode_stays_valid() -> None:
    cfg = {
        "L": 8,
        "ttype": "asmd",
        "value_mod": 23,
        "knowledge_augment": False,
        "bos_token_id": 9999,
    }
    rng = random.Random(3)
    for _ in range(40):
        ex = generate_mano_example(rng, cfg)
        assert validate_mano_example(ex["input_ids"], cfg)


def test_mano_token_override() -> None:
    cfg = {
        "L": 6,
        "ttype": "asm",
        "value_mod": 23,
        "knowledge_augment": False,
        "bos_token_id": 77,
    }
    ex = generate_mano_example(random.Random(9), cfg)
    assert ex["input_ids"][0] == 77
    assert validate_mano_example(ex["input_ids"], cfg)


def test_capo_generation_and_validation() -> None:
    cfg = {
        "num_people": 200,
        "exposures": 5,
        "reverse_md": False,
        "order": 0,
    }
    ds = capo.make_capo_hf_dataset(
        num_examples=20,
        config=cfg,
        seed=4,
        base_dir=ROOT / "Capo-bioS-bioR",
    )
    for row in ds:
        assert isinstance(row["text"], str)
        assert len(row["text"]) > 20
        assert capo.validate_capo_example(row)


def test_lano_generation_and_validation() -> None:
    cfg_path = ROOT / "Lano-cfg" / "configs" / "cfg3f.json"
    cfg = {"config_name": "cfg3f", "config_path": cfg_path}
    rng = random.Random(5)
    for _ in range(3):
        ex = lano.generate_lano_example(rng, cfg)
        assert len(ex["input_ids"]) == len(ex["loss_mask"]) == len(ex["labels"])
        assert lano.validate_lano_example(ex["input_ids"], cfg)

    corrupted = list(ex["input_ids"])
    corrupted[0] = 99
    assert not lano.validate_lano_example(corrupted, cfg)
