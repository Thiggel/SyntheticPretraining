"""Pretrain on Depo and track per-hop evaluation curves.

Run:
  python3 examples/train_depo_curve_hydra.py
  python3 examples/train_depo_curve_hydra.py model=from_pretrained model.name_or_path=gpt2
  python3 examples/train_depo_curve_hydra.py logging.wandb.enabled=true
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Depo.depo import generate_depo_example


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_collate(pad_token_id: int):
    def _collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attention_mask = [], [], []
        for row in batch:
            ids = list(row["input_ids"])
            y = list(row["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad)
            labels.append(y + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return _collate


class DepoIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        config: Dict[str, Any],
        seed: int,
        num_examples: int | None,
    ) -> None:
        self.config = dict(config)
        self.seed = int(seed)
        self.num_examples = num_examples

    def __iter__(self):
        rng = random.Random(self.seed)
        remaining = self.num_examples
        while remaining is None or remaining > 0:
            row = generate_depo_example(rng, self.config)
            yield row
            if remaining is not None:
                remaining -= 1


def _accuracy_on_masked_tokens(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    collate_fn,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits[:, :-1, :]
            labels = batch["labels"][:, 1:]
            mask = labels != -100
            preds = logits.argmax(dim=-1)
            total_correct += int(((preds == labels) & mask).sum().item())
            total_count += int(mask.sum().item())
    model.train()
    return total_correct / max(total_count, 1)


def _build_model_and_tokenizer(cfg: DictConfig):
    if cfg.model.name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=int(cfg.model.vocab_size),
            n_positions=int(cfg.model.n_positions),
            n_ctx=int(cfg.model.n_ctx),
            n_embd=int(cfg.model.n_embd),
            n_layer=int(cfg.model.n_layer),
            n_head=int(cfg.model.n_head),
        )
    )
    return model, None


def _check_vocab_size(model: torch.nn.Module, depo_cfg: Dict[str, int]) -> None:
    needed = max(
        int(depo_cfg["answer_token_id"]),
        int(depo_cfg["query_token_base"]) + int(depo_cfg["K"]),
        int(depo_cfg["separator_token_id"]),
        int(depo_cfg["bos_token_id"]),
        2 * int(depo_cfg["mini_vocab"]),
    )
    vocab_size = int(model.config.vocab_size)
    if vocab_size <= needed:
        raise ValueError(
            f"model vocab_size={vocab_size} is too small for required token id {needed}. "
            "Lower token ids in config or use a larger vocab model."
        )


def _write_metrics_and_plot(
    metrics_rows: List[Dict[str, float]],
    *,
    out_dir: Path,
    metrics_file: str,
    plot_file: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / metrics_file
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "examples_seen", "hop", "accuracy"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    by_hop = defaultdict(list)
    for row in metrics_rows:
        by_hop[int(row["hop"])].append(row)

    plt.figure(figsize=(9, 5))
    for hop in sorted(by_hop):
        points = sorted(by_hop[hop], key=lambda x: x["examples_seen"])
        x = [p["examples_seen"] for p in points]
        y = [p["accuracy"] for p in points]
        plt.plot(x, y, marker="o", label=f"{hop}-hop")
    plt.xlabel("Training examples seen")
    plt.ylabel("Answer-token accuracy")
    plt.title("Depo learning curve by hop length")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plot_path = out_dir / plot_file
    plt.savefig(plot_path, dpi=180)
    plt.close()

    return csv_path, plot_path


def _build_eval_sets(cfg: DictConfig, depo_train_cfg: Dict[str, Any], eval_hops: List[int], step: int):
    eval_sets = {}
    for hop in eval_hops:
        eval_cfg = dict(depo_train_cfg)
        eval_cfg["fixed_k"] = hop
        eval_cfg["qa"] = bool(cfg.eval.qa)
        seed = int(cfg.seed) + int(cfg.eval.seed_offset) + hop
        if bool(cfg.eval.resample_each_eval):
            seed += step * int(cfg.eval.seed_stride)
        eval_sets[hop] = DepoIterableDataset(
            config=eval_cfg,
            seed=seed,
            num_examples=int(cfg.eval.num_examples_per_hop),
        )
    return eval_sets


@hydra.main(version_base=None, config_path="../conf/depo_curve", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))

    model, tokenizer = _build_model_and_tokenizer(cfg)

    depo_train_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if tokenizer is not None:
        bos = getattr(tokenizer, "bos_token_id", None)
        sep = getattr(tokenizer, "sep_token_id", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        if bos is not None:
            depo_train_cfg["bos_token_id"] = int(bos)
        if sep is not None:
            depo_train_cfg["separator_token_id"] = int(sep)
        if eos is not None and "answer_token_id" not in depo_train_cfg:
            depo_train_cfg["answer_token_id"] = int(eos)

    if tokenizer is not None and tokenizer.pad_token_id is not None:
        pad_token_id = int(tokenizer.pad_token_id)
    else:
        pad_token_id = int(depo_train_cfg["bos_token_id"])

    model.config.bos_token_id = int(depo_train_cfg["bos_token_id"])
    model.config.eos_token_id = int(depo_train_cfg["answer_token_id"])
    _check_vocab_size(model, depo_train_cfg)

    device = _resolve_device(str(cfg.train.device))
    model.to(device)

    train_num_examples = int(cfg.train.num_examples)
    train_ds = DepoIterableDataset(
        config=depo_train_cfg,
        seed=int(cfg.seed),
        num_examples=train_num_examples if train_num_examples > 0 else None,
    )

    eval_hops = [int(h) for h in cfg.eval.hops]
    for hop in eval_hops:
        if hop > int(depo_train_cfg["K"]):
            raise ValueError(f"eval hop {hop} exceeds dataset K={depo_train_cfg['K']}")

    eval_sets = {}
    if not bool(cfg.eval.resample_each_eval):
        eval_sets = _build_eval_sets(cfg, depo_train_cfg, eval_hops, step=0)

    collate_fn = _make_collate(pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        drop_last=False,
    )
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    wandb_run = None
    wandb_module = None
    if bool(cfg.logging.wandb.enabled):
        import wandb as wandb_module

        wandb_run = wandb_module.init(
            project=str(cfg.logging.wandb.project),
            entity=cfg.logging.wandb.entity,
            name=cfg.logging.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    metrics_rows: List[Dict[str, float]] = []
    out_dir = Path(cfg.output_dir)
    grad_acc = max(1, int(cfg.train.gradient_accumulation_steps))
    eval_every = max(1, int(cfg.train.eval_every_steps))
    max_steps = int(cfg.train.max_steps)
    examples_seen = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, max_steps + 1):
        running_loss = 0.0
        for _ in range(grad_acc):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            examples_seen += int(batch["input_ids"].size(0))
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_acc
            running_loss += float(out.loss.item())
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if wandb_run is not None:
            wandb_run.log({"train/loss": running_loss, "train/examples_seen": examples_seen}, step=step)

        should_eval = step == 1 or step % eval_every == 0 or step == max_steps
        if not should_eval:
            continue

        if bool(cfg.eval.resample_each_eval):
            eval_sets = _build_eval_sets(cfg, depo_train_cfg, eval_hops, step=step)

        for hop in eval_hops:
            acc = _accuracy_on_masked_tokens(
                model,
                eval_sets[hop],
                batch_size=int(cfg.eval.batch_size),
                collate_fn=collate_fn,
                device=device,
            )
            row = {
                "step": float(step),
                "examples_seen": float(examples_seen),
                "hop": float(hop),
                "accuracy": float(acc),
            }
            metrics_rows.append(row)
            print(
                f"step={step:04d} examples_seen={examples_seen} "
                f"hop={hop} accuracy={acc:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/hop_{hop}_accuracy": acc,
                        "eval/examples_seen": examples_seen,
                    },
                    step=step,
                )

        csv_path, plot_path = _write_metrics_and_plot(
            metrics_rows,
            out_dir=out_dir,
            metrics_file=str(cfg.metrics_file),
            plot_file=str(cfg.plot_file),
        )

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log({"plot/depo_hop_curve": wandb_module.Image(str(plot_path))})
        wandb_run.finish()

    print(f"saved_metrics={csv_path}")
    print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    main()
