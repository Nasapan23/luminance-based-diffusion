# ComfyUI Inference Setup

## 1) Install Official ComfyUI (UI + API)

This repo provides scripts that clone and install the official ComfyUI project:
- `scripts/setup_comfyui.ps1`
- `scripts/setup_comfyui.sh`

They clone:
- `https://github.com/comfyanonymous/ComfyUI`

into `external/ComfyUI` and create `.venv` with required packages.

Windows 11 + AMD Radeon ROCm path:
- AMD currently documents `RX 7600` as supported on Windows for ROCm 7.2.1 PyTorch inference.
- Use the PowerShell installer with the AMD wheel path:

```powershell
scripts/setup_comfyui.ps1 -TorchBackend rocm-windows -PythonExe "C:\Path\To\Python312\python.exe"
```

- ROCm on Windows currently requires Python `3.12`.
- AMD documents Adrenalin driver `26.2.2` for the ROCm 7.2.1 Windows PyTorch release.
- AMD currently documents Windows as inference-only for Radeon ROCm; training remains a Linux-first path.

Run server (includes web UI and API):
- `scripts/run_comfyui.ps1`
- `scripts/run_comfyui.sh`

Default URL:
- `http://127.0.0.1:8188`

For lower-VRAM AMD cards, AMD recommends trying:

```powershell
scripts/run_comfyui.ps1 -AmdDefaults
```

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

Recommended stage split:
- `graygen`: grayscale SDXL checkpoint + grayscale/amphora LoRA
- `recolor`: color-capable SDXL checkpoint for prompt-guided img2img recolor
- `refine`: optional stronger color/refiner checkpoint

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

## 5) Chained Prompt Recolor Pipeline

This repo also supports:

```bash
prompt -> graygen -> recolor -> refine
```

with automatic passing of stage outputs into the next stage:

```bash
python -m lbd.cli infer pipeline \
  --gray-config configs/infer_graygen_comfyui.yaml \
  --recolor-config configs/infer_recolor_comfyui.yaml \
  --refine-config configs/infer_refine_comfyui.yaml
```

Shell wrapper:

```bash
scripts/run_infer_pipeline.sh
```
