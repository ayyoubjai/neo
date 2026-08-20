$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $scriptDir "agi_code.py"

$pythonExe = $env:PYTHON
if (-not $pythonExe) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $pythonExe = $cmd.Source
    } else {
        $cmd = Get-Command py -ErrorAction SilentlyContinue
        if ($cmd) {
            $pythonExe = $cmd.Source
        }
    }
}

if (-not $pythonExe) {
    Write-Error "[agi-code] Python not found. Set PYTHON env var or install Python."
    exit 1
}

& $pythonExe $scriptPath @args
exit $LASTEXITCODE
