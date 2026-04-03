$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

python -m lbd.cli infer refine --config "$repoRoot/configs/infer_refine_comfyui.yaml" $args
