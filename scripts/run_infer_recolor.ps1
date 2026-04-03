$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

python -m lbd.cli infer recolor --config "$repoRoot/configs/infer_recolor_comfyui.yaml" $args
