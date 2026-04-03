#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_CHANNEL="${TORCH_CHANNEL:-auto}" # auto | cu121 | cu124 | cpu
RECREATE_VENV="${RECREATE_VENV:-0}"    # 1 to recreate
SKIP_XFORMERS="${SKIP_XFORMERS:-0}"    # 1 to skip xformers

VENV_PATH="$ROOT/$VENV_DIR"

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_PATH" ]]; then
  rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_EXE" -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel

cuda_detected=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    cuda_detected=1
  fi
fi

if [[ "$TORCH_CHANNEL" == "auto" ]]; then
  if [[ "$cuda_detected" == "1" ]]; then
    TORCH_CHANNEL="cu121"
  else
    TORCH_CHANNEL="cpu"
  fi
fi

case "$TORCH_CHANNEL" in
  cu121) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
  cu124) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124" ;;
  cpu) TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
  *)
    echo "Invalid TORCH_CHANNEL: $TORCH_CHANNEL (use auto|cu121|cu124|cpu)" >&2
    exit 1
    ;;
esac

echo "Installing PyTorch from channel: $TORCH_CHANNEL"
if ! pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"; then
  if [[ "$TORCH_CHANNEL" != "cpu" ]]; then
    echo "CUDA PyTorch install failed. Falling back to CPU wheels." >&2
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
  else
    exit 1
  fi
fi

cd "$ROOT"
pip install -e ".[dev,train]"

if [[ "$SKIP_XFORMERS" != "1" ]]; then
  if ! pip install xformers; then
    echo "xformers install failed. Continuing without xformers." >&2
  fi
fi

echo ""
echo "Bootstrap complete."
echo "Activate venv:"
echo "  source \"$VENV_PATH/bin/activate\""
echo ""
echo "Verify runtime:"
echo "  python -c \"import torch; print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\""

