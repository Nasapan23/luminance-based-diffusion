#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-external/ComfyUI}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_PATH="$ROOT/$TARGET_DIR"

if [[ ! -d "$COMFY_PATH" ]]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_PATH"
fi

cd "$COMFY_PATH"
if [[ ! -d ".venv" ]]; then
  "$PYTHON_EXE" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "ComfyUI setup complete at: $COMFY_PATH"
echo "Run with: scripts/run_comfyui.sh"
