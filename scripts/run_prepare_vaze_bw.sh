#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

INGEST_CFG="$ROOT/configs/sources_vaze.yaml"
INDEX_CSV="$ROOT/data/vaze_bw/meta/ingest_index.csv"
BUILD_CFG="$ROOT/configs/build_vaze_bw.auto.yaml"
OUTPUT_ROOT="$ROOT/data/vaze_bw"

# Optional cap for faster experiments. 0 means "use all valid images".
VAZE_MAX_IMAGES="${VAZE_MAX_IMAGES:-0}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_ts="$(date +%s)"
log "START: amphora prep pipeline"
log "Config: VAZE_MAX_IMAGES=$VAZE_MAX_IMAGES"

log "Step 1/4: ingest local data/vaze images"
python -m lbd.cli data ingest --config "$INGEST_CFG"
log "READY 1/4: ingest finished"

log "Step 2/4: generate auto build config"
python - "$INDEX_CSV" "$BUILD_CFG" "$OUTPUT_ROOT" "$VAZE_MAX_IMAGES" <<'PY'
import csv
import sys
from pathlib import Path

import yaml


index_csv = Path(sys.argv[1])
build_cfg = Path(sys.argv[2])
output_root = Path(sys.argv[3])
max_images = int(sys.argv[4])

if not index_csv.exists():
    raise FileNotFoundError(f"Ingest index not found: {index_csv}")

with index_csv.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

valid_rows = [r for r in rows if str(r.get("valid", "0")).strip().lower() in {"1", "true", "yes", "y"}]
valid_count = len(valid_rows)
if valid_count == 0:
    raise RuntimeError("No valid images found in data/vaze. Put amphora images under data/vaze first.")

target_total = valid_count if max_images <= 0 else min(valid_count, max_images)

if target_total >= 20:
    train = int(target_total * 0.90)
    val = max(1, int(target_total * 0.05))
    test = target_total - train - val
    if test < 1:
        test = 1
        train = max(1, train - 1)
else:
    if target_total == 1:
        train, val, test = 1, 0, 0
    elif target_total == 2:
        train, val, test = 1, 1, 0
    else:
        train, val, test = target_total - 2, 1, 1

config = {
    "run_name": "vaze_bw",
    "runs_root": "runs",
    "input_index_csv": "data/vaze_bw/meta/ingest_index.csv",
    "output_root": "data/vaze_bw",
    "target_total": int(target_total),
    "split_counts": {"train": int(train), "val": int(val), "test": int(test)},
    "seed": 42,
    "threads": 16,
    "batch_size": 256,
    "copy_mode": "copy",
    "dedupe_by_sha256": True,
}

build_cfg.parent.mkdir(parents=True, exist_ok=True)
with build_cfg.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"Prepared build config: {build_cfg}")
print(f"Valid images={valid_count} | target_total={target_total} | split={config['split_counts']}")
PY
log "READY 2/4: build config generated -> $BUILD_CFG"

log "Step 3/4: build color+grayscale amphora dataset"
python -m lbd.cli data build-base100k --config "$BUILD_CFG"
log "READY 3/4: dataset build finished"

log "Step 4/4: write metadata.jsonl for LoRA training"
python - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path


output_root = Path(sys.argv[1])
index_csv = output_root / "meta" / "index.csv"
train_dir = output_root / "gray" / "train"
meta_jsonl = train_dir / "metadata.jsonl"

if not index_csv.exists():
    raise FileNotFoundError(f"Missing build index: {index_csv}")
if not train_dir.exists():
    raise FileNotFoundError(f"Missing train dir: {train_dir}")

rows_out = []
with index_csv.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("split") != "train":
            continue
        if row.get("status") == "failed":
            continue
        image_name = Path(row["gray_path"]).name
        caption = (row.get("caption") or "").strip() or "amphora ceramic vase"
        rows_out.append({"file_name": image_name, "text": caption})

if not rows_out:
    raise RuntimeError("No train rows found to create metadata.jsonl")

with meta_jsonl.open("w", encoding="utf-8") as handle:
    for item in rows_out:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")

print(f"Wrote {len(rows_out)} rows -> {meta_jsonl}")
PY
log "READY 4/4: metadata.jsonl generated"

echo ""
elapsed="$(( $(date +%s) - start_ts ))"
log "READY: amphora prep pipeline finished in ${elapsed}s"
echo "Gray train dir: data/vaze_bw/gray/train"
echo "Caption file: data/vaze_bw/gray/train/metadata.jsonl"
echo "Next: bash scripts/run_train_lora_vaze.sh --dry-run"
