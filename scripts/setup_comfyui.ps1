$ErrorActionPreference = "Stop"

param(
  [string]$TargetDir = "external/ComfyUI",
  [string]$PythonExe = "python",
  [ValidateSet("default", "rocm-windows")]
  [string]$TorchBackend = "default"
)

function Get-PythonVersion {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
  )

  return (& $PythonPath -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')")
}

function Install-RocmWindowsTorch {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
  )

  $pythonVersion = Get-PythonVersion -PythonPath $PythonPath
  if (!$pythonVersion.StartsWith("3.12.")) {
    throw (
      "AMD ROCm on Windows currently requires Python 3.12. " +
      "The ComfyUI venv is using Python $pythonVersion. " +
      "Install Python 3.12 and re-run with -PythonExe pointing to that interpreter."
    )
  }

  $rocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
  $rocmRuntimeWheels = @(
    "$rocmBase/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "$rocmBase/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "$rocmBase/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "$rocmBase/rocm-7.2.1.tar.gz"
  )
  $torchWheels = @(
    "$rocmBase/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "$rocmBase/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "$rocmBase/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
  )

  & $PythonPath -m pip install --no-cache-dir @rocmRuntimeWheels
  & $PythonPath -m pip install --no-cache-dir @torchWheels
}

$repoRoot = Resolve-Path "$PSScriptRoot/.."
$comfyPath = Join-Path $repoRoot $TargetDir
$venvPython = Join-Path $comfyPath ".venv\Scripts\python.exe"

if (!(Test-Path $comfyPath)) {
  git clone https://github.com/comfyanonymous/ComfyUI.git $comfyPath
}

Push-Location $comfyPath
try {
  if (!(Test-Path ".venv")) {
    & $PythonExe -m venv .venv
  }

  & $venvPython -m pip install --upgrade pip

  if ($TorchBackend -eq "rocm-windows") {
    Install-RocmWindowsTorch -PythonPath $venvPython
  }

  & $venvPython -m pip install -r requirements.txt

  Write-Host "ComfyUI setup complete at: $comfyPath"
  Write-Host "Run with: scripts\\run_comfyui.ps1"
  if ($TorchBackend -eq "rocm-windows") {
    Write-Host "AMD ROCm on Windows uses PyTorch inference support only."
    Write-Host "For lower-VRAM cards, try: scripts\\run_comfyui.ps1 -- --lowvram --disable-pinned-memory"
  }
}
finally {
  Pop-Location
}
