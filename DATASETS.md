# Synthetic Task Generators (HF-Ready)

This workspace now exposes runnable, self-contained task generators for:

- `Depo` (reasoning depth)
- `Brevo` (reasoning breadth)
- `Mano` (knowledge manipulation)
- `Capo` (knowledge capacity, bioS-style text)
- `Lano` (hierarchical CFG structure)

## Quick sample commands

```bash
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/inspect_dataset.py
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/inspect_dataset.py dataset=brevo show=2
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/inspect_dataset.py dataset=mano show=2
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/inspect_dataset.py dataset=lano show=2
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/inspect_dataset.py dataset=capo show=2
```

Inspector Hydra configs live in:

- `/Users/filipe/Desktop/synthetic_pretraining/conf/inspect_dataset/config.yaml`
- `/Users/filipe/Desktop/synthetic_pretraining/conf/inspect_dataset/dataset/*.yaml`

## Programmatic usage

Each task script exposes:

- `generate_*_example(...)`
- `validate_*_example(...)`
- `make_*_hf_dataset(num_examples, config, seed)`

All `make_*_hf_dataset(...)` functions return a Hugging Face `datasets.Dataset` (or a small list fallback only if `datasets` is unavailable).

Configs are plain dictionaries now (no dataclasses). Pass config dictionaries directly from Hydra (or define them inline in Python/CLI).

Rows include at least:

- `input_ids` or `text`
- `loss_mask`
- `labels` (for token tasks; built with `-100` where masked)

## Minimal training example

```bash
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/train_depo_minimal.py
```

This trains a tiny GPT-2 model for a few steps on Depo and reports answer-token accuracy.

## Depo Hop-Learning Curves (Hydra)

```bash
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/train_depo_curve_hydra.py
```

This periodically evaluates fixed-hop Depo datasets and saves:

- CSV metrics (`step`, `examples_seen`, `hop`, `accuracy`)
- a multi-line plot (one line per hop)

Hydra configs live in:

- `/Users/filipe/Desktop/synthetic_pretraining/conf/depo_curve/config.yaml`
- `/Users/filipe/Desktop/synthetic_pretraining/conf/depo_curve/dataset/depo1.yaml`
- `/Users/filipe/Desktop/synthetic_pretraining/conf/depo_curve/model/*.yaml`

Example overrides:

```bash
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/train_depo_curve_hydra.py train.max_steps=400 'eval.hops=[1,2,3,4,8,16]' dataset.K=16
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/train_depo_curve_hydra.py model=from_pretrained model.name_or_path=gpt2
python3 /Users/filipe/Desktop/synthetic_pretraining/examples/train_depo_curve_hydra.py logging.wandb.enabled=true logging.wandb.project=my-project
```
