$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

& python -m lbd.cli data download-real-subset --config "$repoRoot/configs/download_real_1k.yaml"
if ($LASTEXITCODE -ne 0) { throw "download-real-subset failed ($LASTEXITCODE)" }

& python -m lbd.cli data ingest --config "$repoRoot/configs/sources_real_1k.yaml"
if ($LASTEXITCODE -ne 0) { throw "ingest failed ($LASTEXITCODE)" }

& python -m lbd.cli data build-base100k --config "$repoRoot/configs/build_base1k_real.yaml"
if ($LASTEXITCODE -ne 0) { throw "build-base100k failed ($LASTEXITCODE)" }

& python -m lbd.cli train sdxl --config "$repoRoot/configs/train_sdxl_real_1k.yaml" --dry-run
if ($LASTEXITCODE -ne 0) { throw "train dry-run failed ($LASTEXITCODE)" }
