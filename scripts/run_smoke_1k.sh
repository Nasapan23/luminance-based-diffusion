#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m lbd.tools.generate_smoke_dataset --repo-root "$ROOT" --per-source 400 --width 128 --height 128
python -m lbd.cli data ingest --config "$ROOT/configs/sources_smoke_1k.yaml"
python -m lbd.cli data build-base100k --config "$ROOT/configs/build_base1k_smoke.yaml"
python -m lbd.cli train sdxl --config "$ROOT/configs/train_sdxl_smoke_1k.yaml" --dry-run
