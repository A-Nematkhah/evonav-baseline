# Run EvoNav with system Python (e.g. torch 2.5+cu121) without activating .venv.
# .venv is used only as a library path: gym 0.15.7, baselines, rvo2.pyd.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$EnvRoot = Split-Path $PSScriptRoot -Parent
$RepoRoot = Split-Path $EnvRoot -Parent
$SitePackages = Join-Path $EnvRoot ".venv\Lib\site-packages"
$VenvRoot = Join-Path $EnvRoot ".venv"
$Baselines = Join-Path $RepoRoot "baselines_openai"

foreach ($p in @($SitePackages, $VenvRoot, $Baselines)) {
    if (-not (Test-Path $p)) {
        Write-Error "Missing path: $p"
    }
}

$env:PYTHONPATH = ($SitePackages, $VenvRoot, $Baselines) -join ";"
Set-Location $EnvRoot
& python (Join-Path $PSScriptRoot "run_evonav.py") @Args
exit $LASTEXITCODE
