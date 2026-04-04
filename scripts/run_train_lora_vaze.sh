#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_ts="$(date +%s)"
log "START: amphora LoRA training"
log "Config: $ROOT/configs/train_lora_vaze.yaml"

if ! python -c "import peft" >/dev/null 2>&1; then
  log "Missing dependency detected: peft. Installing..."
  python -m pip install "peft>=0.11"
  log "READY: peft installed"
fi

python -m lbd.cli train lora --config "$ROOT/configs/train_lora_vaze.yaml" "$@"

elapsed="$(( $(date +%s) - start_ts ))"
log "READY: amphora LoRA command finished in ${elapsed}s"
