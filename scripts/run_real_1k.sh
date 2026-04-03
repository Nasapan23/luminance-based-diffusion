#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m lbd.cli data download-real-subset --config "$ROOT/configs/download_real_1k.yaml"
python -m lbd.cli data ingest --config "$ROOT/configs/sources_real_1k.yaml"
python -m lbd.cli data build-base100k --config "$ROOT/configs/build_base1k_real.yaml"
python -m lbd.cli train sdxl --config "$ROOT/configs/train_sdxl_real_1k.yaml" --dry-run
