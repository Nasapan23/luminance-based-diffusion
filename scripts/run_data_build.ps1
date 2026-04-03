$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

python -m lbd.cli data ingest --config "$repoRoot/configs/sources.yaml"
python -m lbd.cli data build-base100k --config "$repoRoot/configs/build_base100k.yaml"
