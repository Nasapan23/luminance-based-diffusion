# ComfyUI Inference Setup

## 1) Install Official ComfyUI (UI + API)

This repo provides scripts that clone and install the official ComfyUI project:
- `scripts/setup_comfyui.ps1`
- `scripts/setup_comfyui.sh`

They clone:
- `https://github.com/comfyanonymous/ComfyUI`

into `external/ComfyUI` and create `.venv` with required packages.

Run server (includes web UI and API):
- `scripts/run_comfyui.ps1`
- `scripts/run_comfyui.sh`

Default URL:
- `http://127.0.0.1:8188`

If you change host/port, update `comfyui.base_url` in inference configs.

## 2) Place Model Assets in ComfyUI

Place/check model files in ComfyUI model folders:
- `models/checkpoints/` for SDXL checkpoint
- `models/loras/` for LoRA weights

Then set:
- `model.checkpoint_name`
- `model.lora_name`

in:
- `configs/infer_graygen_comfyui.yaml`
- `configs/infer_recolor_comfyui.yaml`
- `configs/infer_refine_comfyui.yaml`

## 3) Configure Stage Inputs

- `graygen`: no input image needed.
- `recolor`: set `defaults.input_image` (or per-job `input_image`) to grayscale structure image.
- `refine`: set `defaults.input_image` (or per-job `input_image`) to recolored image.

## 4) Run

Dry-run first:

```bash
python -m lbd.cli infer graygen --config configs/infer_graygen_comfyui.yaml --dry-run
```

Then full run:

```bash
python -m lbd.cli infer graygen --config configs/infer_graygen_comfyui.yaml
```

Repeat for `recolor` and `refine`.
