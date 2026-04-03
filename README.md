# luminance-based-diffusion

Preparation and orchestration repository for:
- multi-source dataset ingestion (COCO / Places2 / OpenImages adapters)
- RGB + grayscale (3-channel replicated) dataset build
- resumable preprocessing with SQLite history
- SDXL and LoRA training launch orchestration
- ComfyUI-based inference orchestration (graygen / recolor / refine)
- official ComfyUI setup/run scripts (UI + API)

## Quick Start

One-command environment bootstrap (creates `.venv`, installs CUDA PyTorch with CPU fallback, installs all project deps):

```bash
scripts/bootstrap_env.ps1
# or
scripts/bootstrap_env.sh
```

Then activate venv and run:

```bash
python -m lbd.cli data ingest --config configs/sources.yaml
python -m lbd.cli data build-base100k --config configs/build_base100k.yaml
python -m lbd.cli data resume --run-id <run_id>
python -m lbd.cli train sdxl --config configs/train_sdxl.yaml --dry-run
python -m lbd.cli train lora --config configs/train_lora.yaml --dry-run
python -m lbd.cli infer graygen --config configs/infer_graygen_comfyui.yaml --dry-run
python -m lbd.cli infer recolor --config configs/infer_recolor_comfyui.yaml --dry-run
python -m lbd.cli infer refine --config configs/infer_refine_comfyui.yaml --dry-run
scripts/run_smoke_1k.ps1
scripts/run_real_1k.ps1
```

## Notes

- Large dataset/checkpoint content is intentionally git-ignored under `data/` and `runs/`.
- The CLI supports tiny local smoke runs and full runs on stronger training hardware.
- Inference runs write resolved workflows and outputs under `runs/infer/<run_id>/`.
- Training runtime policy prefers CUDA automatically and falls back to CPU when CUDA is unavailable.
- `run_smoke_1k.*` uses synthetic images by design; use `run_real_1k.*` for real-data subset checks.
- For bootstrap overrides: `-TorchChannel cu124` on PowerShell or `TORCH_CHANNEL=cu124 scripts/bootstrap_env.sh`.
