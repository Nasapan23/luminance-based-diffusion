$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

python -m lbd.tools.generate_smoke_dataset --repo-root "$repoRoot" --per-source 400 --width 128 --height 128
python -m lbd.cli data ingest --config "$repoRoot/configs/sources_smoke_1k.yaml"
python -m lbd.cli data build-base100k --config "$repoRoot/configs/build_base1k_smoke.yaml"
python -m lbd.cli train sdxl --config "$repoRoot/configs/train_sdxl_smoke_1k.yaml" --dry-run
