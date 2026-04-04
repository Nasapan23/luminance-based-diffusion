#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_ts="$(date +%s)"

log "START: real 1k pipeline"
log "Step 1/4: download real subset"
python -m lbd.cli data download-real-subset --config "$ROOT/configs/download_real_1k.yaml"
log "READY 1/4: download real subset"

log "Step 2/4: ingest sources"
python -m lbd.cli data ingest --config "$ROOT/configs/sources_real_1k.yaml"
log "READY 2/4: ingest sources"

log "Step 3/4: build grayscale/color dataset"
python -m lbd.cli data build-base100k --config "$ROOT/configs/build_base1k_real.yaml"
log "READY 3/4: build dataset"

log "Step 4/4: SDXL train dry-run"
python -m lbd.cli train sdxl --config "$ROOT/configs/train_sdxl_real_1k.yaml" --dry-run
log "READY 4/4: SDXL train dry-run"

elapsed="$(( $(date +%s) - start_ts ))"
log "READY: real 1k pipeline finished in ${elapsed}s"
