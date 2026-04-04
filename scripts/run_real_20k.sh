#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

bash "$ROOT/scripts/setup_diffusers_examples.sh"
python -m lbd.cli data download-real-subset --config "$ROOT/configs/download_real_20k.yaml"
python -m lbd.cli data ingest --config "$ROOT/configs/sources.yaml"
python -m lbd.cli data build-base100k --config "$ROOT/configs/build_base100k.yaml"
python -m lbd.cli train sdxl --config "$ROOT/configs/train_sdxl.yaml" --dry-run
