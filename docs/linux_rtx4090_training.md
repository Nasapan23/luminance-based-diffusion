# Linux RTX 4090 Training Setup

This is the recommended end-to-end setup for Ubuntu + NVIDIA RTX 4090.

## 1) Prepare System Packages

```bash
sudo apt update
sudo apt install -y git curl wget build-essential python3 python3-venv python3-pip
```

Verify NVIDIA driver and GPU are visible:

```bash
nvidia-smi
```

## 2) Clone Repo

```bash
git clone <your-repo-url> luminance-based-diffusion
cd luminance-based-diffusion
```

## 3) Bootstrap Python + PyTorch (CUDA)

For RTX 4090, prefer CUDA 12.4 wheels:

```bash
TORCH_CHANNEL=cu124 scripts/bootstrap_env.sh
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 4) Pull Diffusers Example Training Scripts

Training configs in this repo point to `external/diffusers/examples/...`.

```bash
bash scripts/setup_diffusers_examples.sh
```

## 5) Download + Build the 20k Dataset

Option A (single script):

```bash
bash scripts/run_real_20k.sh
```

Option B (step-by-step):

```bash
python -m lbd.cli data download-real-subset --config configs/download_real_20k.yaml
python -m lbd.cli data ingest --config configs/sources.yaml
python -m lbd.cli data build-base100k --config configs/build_base100k.yaml
python -m lbd.cli train sdxl --config configs/train_sdxl.yaml --dry-run
```

Expected dataset output path:
- `data/base20k/gray/train`
- `data/base20k/color/train`

## 6) Start Real Training

```bash
python -m lbd.cli train sdxl --config configs/train_sdxl.yaml
```

Optional LoRA phase:

```bash
python -m lbd.cli train lora --config configs/train_lora.yaml
```

## 7) Resume Interrupted Dataset Build

```bash
python -m lbd.cli data resume --run-id <run_id>
```

## 8) Useful Runtime Checks

Watch GPU:

```bash
watch -n 1 nvidia-smi
```

Confirm data count:

```bash
find data/base20k/gray/train -type f | wc -l
```
