#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

OUTPUT_ROOT="${1:-$ROOT/data/base20k}"
INDEX_CSV="$OUTPUT_ROOT/meta/index.csv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "START: backfill dataset metadata"
log "Output root: $OUTPUT_ROOT"

python - "$OUTPUT_ROOT" "$INDEX_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path


output_root = Path(sys.argv[1]).resolve()
index_csv = Path(sys.argv[2]).resolve()
color_caption_overrides = {
    "amphora": "x23_amphoras",
}


def metadata_caption(row: dict[str, str], path_key: str) -> str:
    text = str(row.get("caption", "")).strip()
    if path_key != "color_path":
        return text

    source_id = str(row.get("source_id", "")).strip()
    return color_caption_overrides.get(source_id, text)


if not index_csv.exists():
    raise FileNotFoundError(f"Missing dataset index: {index_csv}")

with index_csv.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

train_rows = [row for row in rows if row.get("split") == "train" and row.get("status") != "failed"]
if not train_rows:
    raise RuntimeError("No successful train rows found in dataset index.")

targets = [
    (output_root / "gray" / "train", "gray_path"),
    (output_root / "color" / "train", "color_path"),
]

for image_dir, path_key in targets:
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = image_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as out:
        for row in train_rows:
            payload = {
                "file_name": Path(row[path_key]).name,
                "text": metadata_caption(row, path_key),
            }
            out.write(json.dumps(payload, ensure_ascii=True) + "\n")
    print(f"Wrote {metadata_path}")

print(f"Train rows exported: {len(train_rows)}")
PY

log "READY: dataset metadata backfilled"
