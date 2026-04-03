$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

python -m lbd.cli infer graygen --config "$repoRoot/configs/infer_graygen_comfyui.yaml" $args
