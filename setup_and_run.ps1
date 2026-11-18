[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$ConfigPath = "training_model/configs/unified_favit_rtx5060ti.yaml",
    [string]$RunsDir = "experiments_runs",
    [switch]$SetupOnly,
    [switch]$DryRun,
    [switch]$Preview,
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"
$taskRepoRoot = $PSScriptRoot
$taskTrainingDir = Join-Path $taskRepoRoot "training_model"
$taskVenvDir = Join-Path $taskRepoRoot ".venv-training"
$taskVenvPython = Join-Path $taskVenvDir "Scripts/python.exe"
$taskRequirements = Join-Path $taskTrainingDir "requirements.txt"
$taskSetupMarker = Join-Path $taskVenvDir "setup-info.json"
$taskTorchVersion = "2.11.0"
$taskVisionVersion = "0.26.0"
# Official CUDA 12.8 wheel pair: https://pytorch.org/get-started/previous-versions/#v2110
$taskTorchIndex = "https://download.pytorch.org/whl/cu128"

if (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath = Join-Path $taskRepoRoot $ConfigPath
}
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
if (-not [System.IO.Path]::IsPathRooted($RunsDir)) {
    $RunsDir = Join-Path $taskTrainingDir $RunsDir
}
$RunsDir = [System.IO.Path]::GetFullPath($RunsDir)

function Invoke-TaskProgram {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Program $($Arguments -join ' ')"
    }
}

if ($Preview) {
    Write-Host "Environment: $taskVenvDir"
    Write-Host "Bootstrap Python: $PythonExe (64-bit Python 3.10-3.14)"
    Write-Host "Install/reuse torch==$taskTorchVersion torchvision==$taskVisionVersion from $taskTorchIndex"
    Write-Host "Install dependencies: $taskRequirements"
    Write-Host "Config: $ConfigPath"
    Write-Host "Runs and resume state: $RunsDir"
    Write-Host "Check runtime imports."
    if (-not $DryRun) { Write-Host "Verify configured CUDA execution." }
    if (-not $SetupOnly -and -not $DryRun) { Write-Host "Check FF++/Celeb-DF folder layout." }
    if ($SetupOnly) {
        Write-Host "Setup only; training will not start."
    } elseif ($DryRun) {
        Write-Host "List scenario states using run_all_experiments.py --dry-run."
    } else {
        Write-Host "Run all scenarios using run_all_experiments.py (automatic resume)."
    }
    return
}

Push-Location $taskRepoRoot
try {
    $env:PYTHONUNBUFFERED = "1"
    $env:NO_ALBUMENTATIONS_UPDATE = "1"
    $env:MPLBACKEND = "Agg"
    $env:MPLCONFIGDIR = Join-Path $taskRepoRoot ".cache/matplotlib"
    if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $taskRepoRoot ".cache/huggingface" }
    if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $taskRepoRoot ".cache/torch" }

    if (-not (Test-Path -LiteralPath $taskVenvPython -PathType Leaf)) {
        Write-Host "[1/4] Creating isolated Python environment..."
        $taskPythonCheck = "import sys; assert (3,10) <= sys.version_info[:2] <= (3,14), 'Use Python 3.10-3.14'; assert sys.maxsize > 2**32, 'Use 64-bit Python'"
        Invoke-TaskProgram -Program $PythonExe -Arguments @("-c", $taskPythonCheck)
        Invoke-TaskProgram -Program $PythonExe -Arguments @("-m", "venv", $taskVenvDir)
    } else {
        Write-Host "[1/4] Reusing $taskVenvDir"
    }

    $taskFingerprint = "$((Get-FileHash -LiteralPath $taskRequirements -Algorithm SHA256).Hash)|$taskTorchVersion|$taskVisionVersion|$taskTorchIndex"
    $taskInstallNeeded = $true
    if ((Test-Path -LiteralPath $taskSetupMarker -PathType Leaf) -and -not $Reinstall) {
        $taskSavedSetup = Get-Content -LiteralPath $taskSetupMarker -Raw | ConvertFrom-Json
        $taskInstallNeeded = $taskSavedSetup.fingerprint -ne $taskFingerprint
    }

    if ($taskInstallNeeded) {
        Write-Host "[2/4] Installing CUDA PyTorch and project dependencies..."
        Invoke-TaskProgram -Program $taskVenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        $taskTorchArgs = @("-m", "pip", "install", "torch==$taskTorchVersion", "torchvision==$taskVisionVersion", "--index-url", $taskTorchIndex)
        if ($Reinstall) { $taskTorchArgs += "--force-reinstall" }
        Invoke-TaskProgram -Program $taskVenvPython -Arguments $taskTorchArgs
        Invoke-TaskProgram -Program $taskVenvPython -Arguments @("-m", "pip", "install", "-r", $taskRequirements, "torch==$taskTorchVersion", "torchvision==$taskVisionVersion")
        Invoke-TaskProgram -Program $taskVenvPython -Arguments @("-m", "pip", "check")
        $taskPackageSnapshot = & $taskVenvPython -m pip freeze
        if ($LASTEXITCODE -ne 0) { throw "Cannot record installed package versions." }
        $taskPackageSnapshot | Set-Content -LiteralPath (Join-Path $taskVenvDir "installed-packages.txt") -Encoding UTF8
        @{ fingerprint = $taskFingerprint } | ConvertTo-Json | Set-Content -LiteralPath $taskSetupMarker -Encoding UTF8
    } else {
        Write-Host "[2/4] Dependencies unchanged; using the installed versions."
    }

    Write-Host "[3/4] Checking environment and configuration..."
    $taskCheckArgs = @("-u", (Join-Path $taskTrainingDir "check_training_setup.py"), "--config", $ConfigPath)
    if ($SetupOnly -or $DryRun) { $taskCheckArgs += "--skip-data-check" }
    if ($DryRun) { $taskCheckArgs += "--skip-cuda-check" }
    Invoke-TaskProgram -Program $taskVenvPython -Arguments $taskCheckArgs

    if ($SetupOnly) {
        Write-Host "Setup complete. Run this launcher again to start/resume experiments."
    } else {
        Write-Host "[4/4] Starting experiment runner. Existing progress will be reused."
        $taskRunnerArgs = @("-u", (Join-Path $taskTrainingDir "run_all_experiments.py"), "--base-config", $ConfigPath, "--runs-dir", $RunsDir)
        if ($DryRun) { $taskRunnerArgs += "--dry-run" }
        Invoke-TaskProgram -Program $taskVenvPython -Arguments $taskRunnerArgs
    }
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "After correcting the problem, run this launcher again. Progress is kept in $RunsDir"
    exit 1
} finally {
    Pop-Location
}
