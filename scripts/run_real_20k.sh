#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
INGEST_INDEX="$ROOT/data/base20k/meta/ingest_index.csv"
AUTO_BUILD_CFG="$ROOT/configs/build_base20k.auto.yaml"
DOWNLOAD_CFG="${DOWNLOAD_CFG:-$ROOT/configs/download_real_20k_stable.yaml}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_ts="$(date +%s)"

log "START: real 20k pipeline"
log "Download config: $DOWNLOAD_CFG"
log "Step 0/4: setup diffusers examples"
bash "$ROOT/scripts/setup_diffusers_examples.sh"
log "READY 0/4: setup diffusers examples"

log "Step 1/4: download real subset"
python -m lbd.cli data download-real-subset --config "$DOWNLOAD_CFG"
log "READY 1/4: download real subset"

log "Step 2/4: ingest sources"
python -m lbd.cli data ingest --config "$ROOT/configs/sources.yaml"
log "READY 2/4: ingest sources"

log "Step 2.5/4: derive auto build target from available valid images"
python - "$INGEST_INDEX" "$AUTO_BUILD_CFG" <<'PY'
import csv
import sys
from pathlib import Path

import yaml

from lbd.data.build_base100k import _dedupe_by_hash, _is_true


ingest_index = Path(sys.argv[1])
auto_build_cfg = Path(sys.argv[2])
if not ingest_index.exists():
    raise FileNotFoundError(f"Ingest index not found: {ingest_index}")

with ingest_index.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

valid_rows = [r for r in rows if _is_true(r.get("valid", "0"))]
raw_valid_count = len(valid_rows)
if raw_valid_count < 10:
    raise RuntimeError(f"Too few valid images after ingest: {raw_valid_count}")

valid_count = len(_dedupe_by_hash(valid_rows))
if valid_count < 10:
    raise RuntimeError(
        f"Too few valid images after dedupe: {valid_count} "
        f"(raw valid before dedupe: {raw_valid_count})"
    )

target_total = min(valid_count, 20000)

train = int(target_total * 0.8)
val = int(target_total * 0.1)
test = target_total - train - val
if target_total >= 3 and test == 0:
    test = 1
    if train > val:
        train -= 1
    else:
        val = max(0, val - 1)

cfg = {
    "run_name": "base20k_auto",
    "runs_root": "runs",
    "input_index_csv": "data/base20k/meta/ingest_index.csv",
    "output_root": "data/base20k",
    "target_total": int(target_total),
    "split_counts": {"train": int(train), "val": int(val), "test": int(test)},
    "seed": 42,
    "threads": 16,
    "batch_size": 512,
    "copy_mode": "copy",
    "dedupe_by_sha256": True,
}

with auto_build_cfg.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(f"Auto build config written: {auto_build_cfg}")
print(
    f"raw_valid_count={raw_valid_count} "
    f"deduped_valid_count={valid_count} "
    f"target_total={target_total} "
    f"split={cfg['split_counts']}"
)
PY
log "READY 2.5/4: auto build config generated"

log "Step 3/4: build grayscale/color dataset"
python -m lbd.cli data build-base100k --config "$AUTO_BUILD_CFG"
log "READY 3/4: build dataset"

log "Step 4/4: SDXL train dry-run"
python -m lbd.cli train sdxl --config "$ROOT/configs/train_sdxl.yaml" --dry-run
log "READY 4/4: SDXL train dry-run"

elapsed="$(( $(date +%s) - start_ts ))"
log "READY: real 20k pipeline finished in ${elapsed}s"
