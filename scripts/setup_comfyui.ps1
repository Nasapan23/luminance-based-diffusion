$ErrorActionPreference = "Stop"

param(
  [string]$TargetDir = "external/ComfyUI",
  [string]$PythonExe = "python"
)

$repoRoot = Resolve-Path "$PSScriptRoot/.."
$comfyPath = Join-Path $repoRoot $TargetDir

if (!(Test-Path $comfyPath)) {
  git clone https://github.com/comfyanonymous/ComfyUI.git $comfyPath
}

Push-Location $comfyPath
try {
  if (!(Test-Path ".venv")) {
    & $PythonExe -m venv .venv
  }

  & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
  & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

  Write-Host "ComfyUI setup complete at: $comfyPath"
  Write-Host "Run with: scripts\\run_comfyui.ps1"
}
finally {
  Pop-Location
}
