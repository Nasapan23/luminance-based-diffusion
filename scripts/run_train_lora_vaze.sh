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

log "Ensuring compatible training deps (transformers<5, peft<0.12)"
python -m pip install "transformers>=4.41,<5" "peft>=0.11,<0.12"
log "READY: dependency compatibility check complete"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU memory snapshot before training:"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  active_procs="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${active_procs// }" ]]; then
    log "Detected active GPU compute processes:"
    echo "$active_procs"
    log "If run fails immediately, stop other GPU jobs and retry this script."
  fi
fi

python -m lbd.cli train lora --config "$ROOT/configs/train_lora_vaze.yaml" "$@"

elapsed="$(( $(date +%s) - start_ts ))"
log "READY: amphora LoRA command finished in ${elapsed}s"
