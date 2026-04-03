$ErrorActionPreference = "Stop"

param(
  [string]$ComfyDir = "external/ComfyUI",
  [string]$Host = "127.0.0.1",
  [int]$Port = 8188
)

$repoRoot = Resolve-Path "$PSScriptRoot/.."
$comfyPath = Join-Path $repoRoot $ComfyDir
$pythonPath = Join-Path $comfyPath ".venv\\Scripts\\python.exe"

if (!(Test-Path $pythonPath)) {
  throw "ComfyUI venv not found at $pythonPath. Run scripts/setup_comfyui.ps1 first."
}

Push-Location $comfyPath
try {
  & $pythonPath main.py --listen $Host --port $Port $args
}
finally {
  Pop-Location
}
