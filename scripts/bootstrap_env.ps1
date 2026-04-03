param(
  [string]$PythonExe = "python",
  [string]$VenvDir = ".venv",
  [string]$TorchChannel = "auto",
  [switch]$RecreateVenv,
  [switch]$SkipXformers
)

$ErrorActionPreference = "Stop"

function Invoke-Pip {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & $pipExe @Args
}

$repoRoot = Resolve-Path "$PSScriptRoot/.."
$venvPath = Join-Path $repoRoot $VenvDir

if ($RecreateVenv -and (Test-Path $venvPath)) {
  Remove-Item -Recurse -Force $venvPath
}

if (!(Test-Path $venvPath)) {
  & $PythonExe -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\\python.exe"
$pipExe = Join-Path $venvPath "Scripts\\pip.exe"
if (!(Test-Path $pythonExe)) {
  throw "Venv python not found: $pythonExe"
}

Write-Host "Using venv: $venvPath"
& $pythonExe -m pip install --upgrade pip setuptools wheel

$cudaDetected = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  try {
    $null = & nvidia-smi -L
    if ($LASTEXITCODE -eq 0) {
      $cudaDetected = $true
    }
  } catch {
    $cudaDetected = $false
  }
}

if ($TorchChannel -eq "auto") {
  if ($cudaDetected) {
    $TorchChannel = "cu121"
  } else {
    $TorchChannel = "cpu"
  }
}

$torchIndexMap = @{
  "cu121" = "https://download.pytorch.org/whl/cu121"
  "cu124" = "https://download.pytorch.org/whl/cu124"
  "cpu" = "https://download.pytorch.org/whl/cpu"
}

if (!$torchIndexMap.ContainsKey($TorchChannel)) {
  throw "Invalid TorchChannel '$TorchChannel'. Use: auto, cu121, cu124, cpu"
}

$selectedIndex = $torchIndexMap[$TorchChannel]
Write-Host "Installing PyTorch from channel: $TorchChannel"

$torchInstallOk = $true
try {
  Invoke-Pip install torch torchvision torchaudio --index-url $selectedIndex
} catch {
  $torchInstallOk = $false
}

if (!$torchInstallOk -and $TorchChannel -ne "cpu") {
  Write-Warning "CUDA PyTorch install failed. Falling back to CPU wheels."
  Invoke-Pip install torch torchvision torchaudio --index-url $torchIndexMap["cpu"]
}

Push-Location $repoRoot
try {
  & $pythonExe -m pip install -e ".[dev,train]"

  if (-not $SkipXformers) {
    try {
      & $pythonExe -m pip install xformers
    } catch {
      Write-Warning "xformers install failed. Continuing without xformers."
    }
  }
}
finally {
  Pop-Location
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Activate venv:"
Write-Host "  $venvPath\\Scripts\\Activate.ps1"
Write-Host ""
Write-Host "Verify runtime:"
$verifyCmd = '{0} -c "import torch; print(''cuda'', torch.cuda.is_available()); print(''device'', torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''cpu'')"' -f $pythonExe
Write-Host "  $verifyCmd"
