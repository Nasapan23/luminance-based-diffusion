# Runbook

## 1) Environment

One command setup:

```bash
scripts/bootstrap_env.ps1
# or
scripts/bootstrap_env.sh
```

Optional CUDA channel override:
- PowerShell: `scripts/bootstrap_env.ps1 -TorchChannel cu124`
- Bash: `TORCH_CHANNEL=cu124 scripts/bootstrap_env.sh`

Manual alternative:

```bash
python -m pip install -e .[dev,train]
```

Install official ComfyUI (UI + API server):

```bash
scripts/setup_comfyui.ps1
# or
scripts/setup_comfyui.sh
```

## 2) Ingest Sources

Fill dataset files under:
- `data/raw/coco`
- `data/raw/places2`
- `data/raw/openimages`

Then run:

```bash
python -m lbd.cli data ingest --config configs/sources.yaml
```

Output:
- `data/base100k/meta/ingest_index.csv`

## 3) Build 100k Dataset (80k/10k/10k)

```bash
python -m lbd.cli data build-base100k --config configs/build_base100k.yaml
```

Outputs:
- `data/base100k/color/{train,val,test}/...`
- `data/base100k/gray/{train,val,test}/...`
- `data/base100k/meta/index.csv`
- `data/base100k/meta/split_counts.csv`
- `data/base100k/meta/source_contributions.csv`
- `data/base100k/meta/captions.csv`
- `runs/<run_id>/history.sqlite`

## 4) Resume If Interrupted

```bash
python -m lbd.cli data resume --run-id <run_id>
```

Resume will process only non-`done` items and keep full event/error history.

## 5) Training Command Dry-Run

```bash
python -m lbd.cli train sdxl --config configs/train_sdxl.yaml --dry-run
python -m lbd.cli train lora --config configs/train_lora.yaml --dry-run
```

Remove `--dry-run` on the stronger training machine once paths/scripts are confirmed.

## 6) ComfyUI Inference (Graygen / Recolor / Refine)

Dry-run (builds resolved workflows without hitting ComfyUI API):

```bash
python -m lbd.cli infer graygen --config configs/infer_graygen_comfyui.yaml --dry-run
python -m lbd.cli infer recolor --config configs/infer_recolor_comfyui.yaml --dry-run
python -m lbd.cli infer refine --config configs/infer_refine_comfyui.yaml --dry-run
```

Full execution:

```bash
python -m lbd.cli infer graygen --config configs/infer_graygen_comfyui.yaml
python -m lbd.cli infer recolor --config configs/infer_recolor_comfyui.yaml
python -m lbd.cli infer refine --config configs/infer_refine_comfyui.yaml
```

Each run writes:
- `runs/infer/<run_id>/config.resolved.yaml`
- `runs/infer/<run_id>/jobs.resolved.json`
- `runs/infer/<run_id>/workflows/*.json`
- `runs/infer/<run_id>/results.csv`
- `runs/infer/<run_id>/events.csv`
- `runs/infer/<run_id>/outputs/*`

## 7) 1k Smoke E2E Check

This creates synthetic data for three sources, builds a 1k dataset, and validates training launch config:

```bash
scripts/run_smoke_1k.ps1
# or
scripts/run_smoke_1k.sh
```

Configs used:
- `configs/sources_smoke_1k.yaml`
- `configs/build_base1k_smoke.yaml`
- `configs/train_sdxl_smoke_1k.yaml`

Important:
- This is intentionally synthetic test data, not realistic photos.
- Raw input path is `data/raw_smoke/*`.

## 8) 1k Real-Data Subset Check

Use this when you already have real COCO/Places2/OpenImages data under `data/raw/*`:

```bash
scripts/run_real_1k.ps1
# or
scripts/run_real_1k.sh
```

Configs used:
- `configs/sources_real_1k.yaml`
- `configs/build_base1k_real.yaml`
- `configs/train_sdxl_real_1k.yaml`
