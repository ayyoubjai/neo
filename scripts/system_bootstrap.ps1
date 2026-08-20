# System Bootstrap Script
# 
# This is a PowerShell wrapper for the system bootstrap utility.
# It guides you through setting up your AI system's identity and personality.
#
# Usage:
#   On Windows: .\scripts\system_bootstrap.ps1
#   On Linux/Mac: bash scripts/system_bootstrap.sh

param()

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir

Set-Location $rootDir

# Run the Python bootstrap script
python scripts\system_bootstrap.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Bootstrap script failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
