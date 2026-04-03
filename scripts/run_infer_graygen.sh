#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m lbd.cli infer graygen --config "$ROOT/configs/infer_graygen_comfyui.yaml" "$@"
