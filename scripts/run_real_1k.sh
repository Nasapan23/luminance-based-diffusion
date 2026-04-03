#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

if [[ ! -d "$ROOT/data/raw/coco" && ! -d "$ROOT/data/raw/places2" && ! -d "$ROOT/data/raw/openimages" ]]; then
  echo "No real dataset folders found under data/raw/{coco,places2,openimages}. Add real data first." >&2
  exit 1
fi

python -m lbd.cli data ingest --config "$ROOT/configs/sources_real_1k.yaml"
python -m lbd.cli data build-base100k --config "$ROOT/configs/build_base1k_real.yaml"
python -m lbd.cli train sdxl --config "$ROOT/configs/train_sdxl_real_1k.yaml" --dry-run
