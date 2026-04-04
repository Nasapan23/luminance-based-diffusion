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

Pull diffusers example training scripts used by the train configs:

```bash
bash scripts/setup_diffusers_examples.sh
```

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

Linux RTX 4090 focused setup:
- `docs/linux_rtx4090_training.md`

## 2) Download Real Subset (Optional)

If you want this repo to download a starter real dataset automatically:

```bash
python -m lbd.cli data download-real-subset --config configs/download_real_20k.yaml
```

This targets 20,000 total images (roughly balanced across COCO / Places2 / OpenImages).

## 3) Ingest Sources

Fill dataset files under:
- `data/raw/coco`
- `data/raw/places2`
- `data/raw/openimages`

Then run:

```bash
python -m lbd.cli data ingest --config configs/sources.yaml
```

Output:
- `data/base20k/meta/ingest_index.csv`

## 4) Build 20k Dataset (16k/2k/2k)

```bash
python -m lbd.cli data build-base100k --config configs/build_base100k.yaml
```

Outputs:
- `data/base20k/color/{train,val,test}/...`
- `data/base20k/gray/{train,val,test}/...`
- `data/base20k/meta/index.csv`
- `data/base20k/meta/split_counts.csv`
- `data/base20k/meta/source_contributions.csv`
- `data/base20k/meta/captions.csv`
- `runs/<run_id>/history.sqlite`

## 5) Resume If Interrupted

```bash
python -m lbd.cli data resume --run-id <run_id>
```

Resume will process only non-`done` items and keep full event/error history.

## 6) Training Command Dry-Run

```bash
python -m lbd.cli train sdxl --config configs/train_sdxl.yaml --dry-run
python -m lbd.cli train lora --config configs/train_lora.yaml --dry-run
```

Remove `--dry-run` on the stronger training machine once paths/scripts are confirmed.

## 7) ComfyUI Inference (Graygen / Recolor / Refine)

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

## 8) 1k Smoke E2E Check

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

## 9) 1k Real-Data Subset Check

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

## 10) 20k Real-Data One-Command Pipeline (Linux)

```bash
bash scripts/run_real_20k.sh
```

This runs:
- `configs/download_real_20k.yaml`
- `configs/sources.yaml`
- `configs/build_base100k.yaml` (configured to 20k output)
- `configs/train_sdxl.yaml` (`--dry-run`)
