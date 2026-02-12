from __future__ import annotations

import sys
from unittest.mock import patch
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import RecurrentOlmo3ForCausalLM


def _base_cfg() -> dict:
    return {
        "hf": {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "attention_dropout": 0.0,
            "attention_bias": False,
            "use_cache": False,
        },
        "recurrent": {
            "num_loops": 3,
            "encoder_layers": 1,
            "loop_layers": 2,
            "decoder_layers": 1,
            "inject_input_each_step": False,
            "loop_attention_mode": "self",
            "random_init_loop_state": False,
            "random_init_std": 0.5,
            "tbptt_steps": 0,
            "train_recursion_mode": "fixed",
            "poisson_mode_offset": 0.001,
            "act": {"enabled": False, "kl_weight": 0.01},
        },
    }


def test_recurrent_olmo3_forward_without_act() -> None:
    torch.manual_seed(0)
    model = RecurrentOlmo3ForCausalLM(_base_cfg())
    input_ids = torch.randint(0, 120, (2, 12), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids)

    assert out.logits.shape == (2, 12, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.act_exit_probs is None
    assert out.step_losses is None


def test_recurrent_olmo3_forward_with_injection_tbptt_and_act() -> None:
    torch.manual_seed(0)
    cfg = _base_cfg()
    cfg["recurrent"]["inject_input_each_step"] = True
    cfg["recurrent"]["random_init_loop_state"] = True
    cfg["recurrent"]["tbptt_steps"] = 2
    cfg["recurrent"]["act"] = {"enabled": True, "kl_weight": 0.05}

    model = RecurrentOlmo3ForCausalLM(cfg)
    input_ids = torch.randint(0, 120, (2, 12), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids)

    assert out.logits.shape == (2, 12, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.act_exit_probs is not None
    assert out.step_losses is not None
    assert out.act_kl_loss is not None

    assert out.act_exit_probs.shape == (3,)
    assert out.step_losses.shape == (3,)
    assert torch.isclose(out.act_exit_probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_recurrent_olmo3_poisson_training_recursions() -> None:
    cfg = _base_cfg()
    cfg["recurrent"]["num_loops"] = 4
    cfg["recurrent"]["train_recursion_mode"] = "poisson"
    cfg["recurrent"]["act"] = {"enabled": True, "kl_weight": 0.05}

    model = RecurrentOlmo3ForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(0, 120, (2, 12), dtype=torch.long)

    with patch("models.recurrent_mixins.torch.poisson", return_value=torch.tensor(2.0)):
        out = model(input_ids=input_ids, labels=input_ids)

    assert out.step_losses is not None
    assert out.act_exit_probs is not None
    assert out.step_losses.shape == (2,)
    assert out.act_exit_probs.shape == (2,)


def test_recurrent_olmo3_cross_loop_attention() -> None:
    cfg = _base_cfg()
    cfg["recurrent"]["loop_attention_mode"] = "cross"
    cfg["recurrent"]["random_init_loop_state"] = True
    cfg["recurrent"]["inject_input_each_step"] = False

    model = RecurrentOlmo3ForCausalLM(cfg)
    input_ids = torch.randint(0, 120, (2, 12), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids)

    assert out.logits.shape == (2, 12, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
