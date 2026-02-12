"""Minimal end-to-end training example on Depo using iterable on-the-fly data.

Run:
  python3 examples/train_depo_minimal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import DepoTask, TaskIterableDataset


def _collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []

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

def evaluate_answer_accuracy(model: GPT2LMHeadModel, rows) -> dict[str, float]:
    model.eval()
    loader = DataLoader(rows, batch_size=8, shuffle=False, collate_fn=_collate)
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

    train_task = DepoTask(train_cfg)
    eval_task = DepoTask(eval_cfg)
    train_ds = TaskIterableDataset(task=train_task, seed=123, num_examples=None)
    eval_rows = eval_task.take(seed=456, count=64)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=_collate)
    train_iter = iter(train_loader)

    for _ in range(10):
        batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    result = evaluate_answer_accuracy(model, eval_rows)
    print(f"eval_loss={result['loss']:.4f}")
    print(f"answer_token_accuracy={result['answer_token_accuracy']:.4f}")

    sample = eval_rows[0]
    is_valid = eval_task.validate_example(sample)
    print(f"first_eval_sample_valid={is_valid}")


if __name__ == "__main__":
    main()
