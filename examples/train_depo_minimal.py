"""Minimal end-to-end training example on Depo using Hugging Face Transformers.

Run:
  python3 examples/train_depo_minimal.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Depo.depo import make_depo_hf_dataset, validate_depo_example


def _collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids: List[List[int]] = []
    labels: List[List[int]] = []
    attention_mask: List[List[int]] = []

    for row in batch:
        ids = list(row["input_ids"])
        y = list(row["labels"])
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        labels.append(y + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

def evaluate_answer_accuracy(model: GPT2LMHeadModel, dataset) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=_collate)
    device = next(model.parameters()).device

    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += float(out.loss.item())
            total_batches += 1

            logits = out.logits[:, :-1, :]
            labels = batch["labels"][:, 1:]
            mask = labels != -100
            preds = logits.argmax(dim=-1)

            total_correct += int(((preds == labels) & mask).sum().item())
            total_count += int(mask.sum().item())

    return {
        "loss": total_loss / max(total_batches, 1),
        "answer_token_accuracy": total_correct / max(total_count, 1),
    }


def main() -> None:
    train_cfg = {
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
    eval_cfg = dict(train_cfg)

    train_ds = make_depo_hf_dataset(num_examples=256, config=train_cfg, seed=123)
    eval_ds = make_depo_hf_dataset(num_examples=64, config=eval_cfg, seed=456)

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10064,
            n_positions=2048,
            n_ctx=2048,
            n_embd=128,
            n_layer=2,
            n_head=4,
            bos_token_id=train_cfg["bos_token_id"],
            eos_token_id=train_cfg["answer_token_id"],
        )
    )

    args = TrainingArguments(
        output_dir=str(ROOT / "tmp_depo_train"),
        max_steps=10,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=_collate,
    )
    trainer.train()

    result = evaluate_answer_accuracy(model, eval_ds)
    print(f"eval_loss={result['loss']:.4f}")
    print(f"answer_token_accuracy={result['answer_token_accuracy']:.4f}")

    sample = eval_ds[0]
    is_valid = validate_depo_example(sample["input_ids"], eval_cfg)
    print(f"first_eval_sample_valid={is_valid}")


if __name__ == "__main__":
    main()
