param(
  [ValidateSet("auto", "cpu", "gpu")]
  [string]$Device = "auto",
  [switch]$QuickTest,
  [switch]$Fp16
)

$Python = ".\.venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
  Write-Error "Missing .venv. Run scripts\setup_environment_windows.ps1 first."
  exit 1
}

$ArgsList = @("scripts\run_prepared_data_pipeline.py", "--device", $Device)
if ($QuickTest) { $ArgsList += "--quick-test" }
if ($Fp16) { $ArgsList += "--fp16" }

& $Python @ArgsList
