$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = "$repoRoot/src"

python -m lbd.cli train lora --config "$repoRoot/configs/train_lora.yaml" $args
