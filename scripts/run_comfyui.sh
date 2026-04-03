#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-external/ComfyUI}"
HOST="${COMFY_HOST:-127.0.0.1}"
PORT="${COMFY_PORT:-8188}"
COMFY_PATH="$ROOT/$COMFY_DIR"

if [[ ! -x "$COMFY_PATH/.venv/bin/python" ]]; then
  echo "ComfyUI venv missing at $COMFY_PATH/.venv. Run scripts/setup_comfyui.sh first." >&2
  exit 1
fi

cd "$COMFY_PATH"
source .venv/bin/activate
python main.py --listen "$HOST" --port "$PORT" "$@"
