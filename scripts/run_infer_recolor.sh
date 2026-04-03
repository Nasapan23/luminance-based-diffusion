#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m lbd.cli infer recolor --config "$ROOT/configs/infer_recolor_comfyui.yaml" "$@"
