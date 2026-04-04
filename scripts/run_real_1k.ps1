$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"
$started = Get-Date

function Log-Stage([string]$Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $Message"
}

Log-Stage "START: real 1k pipeline"
Log-Stage "Step 1/4: download real subset"

& python -m lbd.cli data download-real-subset --config "$repoRoot/configs/download_real_1k.yaml"
if ($LASTEXITCODE -ne 0) { throw "download-real-subset failed ($LASTEXITCODE)" }
Log-Stage "READY 1/4: download real subset"

Log-Stage "Step 2/4: ingest sources"

& python -m lbd.cli data ingest --config "$repoRoot/configs/sources_real_1k.yaml"
if ($LASTEXITCODE -ne 0) { throw "ingest failed ($LASTEXITCODE)" }
Log-Stage "READY 2/4: ingest sources"

Log-Stage "Step 3/4: build grayscale/color dataset"

& python -m lbd.cli data build-base100k --config "$repoRoot/configs/build_base1k_real.yaml"
if ($LASTEXITCODE -ne 0) { throw "build-base100k failed ($LASTEXITCODE)" }
Log-Stage "READY 3/4: build dataset"

Log-Stage "Step 4/4: SDXL train dry-run"

& python -m lbd.cli train sdxl --config "$repoRoot/configs/train_sdxl_real_1k.yaml" --dry-run
if ($LASTEXITCODE -ne 0) { throw "train dry-run failed ($LASTEXITCODE)" }
Log-Stage "READY 4/4: SDXL train dry-run"

$elapsed = [int]((Get-Date) - $started).TotalSeconds
Log-Stage "READY: real 1k pipeline finished in ${elapsed}s"
