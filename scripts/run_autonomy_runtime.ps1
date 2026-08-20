$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot\src;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = "$repoRoot\src"
}

Push-Location $repoRoot
try {
    & $pythonExe -m autonomy.runtime @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
