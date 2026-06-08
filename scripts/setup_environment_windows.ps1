param(
  [ValidateSet("cpu", "gpu")]
  [string]$Device = "gpu",

  [ValidateSet("cu128", "cu126", "cu121", "nightly-cu128", "nightly-cu129")]
  [string]$Cuda = "cu128",

  [switch]$AllowNightlyFallback,
  [switch]$SkipBase
)

$Python = ".\.venv\Scripts\python.exe"

if (!(Test-Path $Python)) {
  Write-Host "Missing .venv. Creating a virtual environment with the default python on PATH..."
  python -m venv .venv
}

$ArgsList = @("scripts\setup_environment.py", "--device", $Device, "--cuda", $Cuda)
if ($AllowNightlyFallback) { $ArgsList += "--allow-nightly-fallback" }
if ($SkipBase) { $ArgsList += "--skip-base" }

& $Python @ArgsList
