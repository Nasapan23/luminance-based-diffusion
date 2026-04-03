$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

if (!(Test-Path "$repoRoot/data/raw/coco") -and !(Test-Path "$repoRoot/data/raw/places2") -and !(Test-Path "$repoRoot/data/raw/openimages")) {
  throw "No real dataset folders found under data/raw/{coco,places2,openimages}. Add real data first."
}

python -m lbd.cli data ingest --config "$repoRoot/configs/sources_real_1k.yaml"
python -m lbd.cli data build-base100k --config "$repoRoot/configs/build_base1k_real.yaml"
python -m lbd.cli train sdxl --config "$repoRoot/configs/train_sdxl_real_1k.yaml" --dry-run
