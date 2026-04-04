#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m lbd.cli infer pipeline \
  --gray-config "$ROOT/configs/infer_graygen_comfyui.yaml" \
  --recolor-config "$ROOT/configs/infer_recolor_comfyui.yaml" \
  --refine-config "$ROOT/configs/infer_refine_comfyui.yaml" \
  "$@"
