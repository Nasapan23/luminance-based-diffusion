#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-$ROOT/external/diffusers}"
DIFFUSERS_REF="${DIFFUSERS_REF:-v0.29.2}"
DIFFUSERS_REPO="${DIFFUSERS_REPO:-https://github.com/huggingface/diffusers.git}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_ts="$(date +%s)"
log "START: setup diffusers examples"
log "Target: ref=$DIFFUSERS_REF dir=$DIFFUSERS_DIR"

if [[ -d "$DIFFUSERS_DIR/.git" ]]; then
  log "Existing diffusers repo found. Fetching and checking out ref."
  git -C "$DIFFUSERS_DIR" fetch --tags origin
  git -C "$DIFFUSERS_DIR" checkout "$DIFFUSERS_REF"
else
  log "Diffusers repo missing. Cloning."
  mkdir -p "$(dirname "$DIFFUSERS_DIR")"
  git clone --depth 1 --branch "$DIFFUSERS_REF" "$DIFFUSERS_REPO" "$DIFFUSERS_DIR"
fi

elapsed="$(( $(date +%s) - start_ts ))"
log "READY: diffusers examples at $DIFFUSERS_DIR (${elapsed}s)"
