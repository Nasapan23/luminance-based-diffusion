#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m lbd.cli data ingest --config "$ROOT/configs/sources.yaml"
python -m lbd.cli data build-base100k --config "$ROOT/configs/build_base100k.yaml"
