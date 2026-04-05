param(
  [string]$ComfyDir = "external/ComfyUI",
  [Alias("Host")]
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8188,
  [switch]$AmdDefaults
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path "$PSScriptRoot/.."
$comfyPath = Join-Path $repoRoot $ComfyDir
$pythonPath = Join-Path $comfyPath ".venv\\Scripts\\python.exe"

if (!(Test-Path $pythonPath)) {
  throw "ComfyUI venv not found at $pythonPath. Run scripts/setup_comfyui.ps1 first."
}

Push-Location $comfyPath
try {
  $launchArgs = @("--listen", $ListenHost, "--port", $Port)
  if ($AmdDefaults) {
    $launchArgs += @("--lowvram", "--disable-pinned-memory")
  }
  $launchArgs += $args
  & $pythonPath main.py @launchArgs
}
finally {
  Pop-Location
}
