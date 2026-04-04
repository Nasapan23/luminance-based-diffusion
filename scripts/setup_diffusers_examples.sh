#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-$ROOT/external/diffusers}"
DIFFUSERS_REF="${DIFFUSERS_REF:-v0.29.2}"
DIFFUSERS_REPO="${DIFFUSERS_REPO:-https://github.com/huggingface/diffusers.git}"

if [[ -d "$DIFFUSERS_DIR/.git" ]]; then
  git -C "$DIFFUSERS_DIR" fetch --tags origin
  git -C "$DIFFUSERS_DIR" checkout "$DIFFUSERS_REF"
else
  mkdir -p "$(dirname "$DIFFUSERS_DIR")"
  git clone --depth 1 --branch "$DIFFUSERS_REF" "$DIFFUSERS_REPO" "$DIFFUSERS_DIR"
fi

echo "Diffusers examples ready at: $DIFFUSERS_DIR"
