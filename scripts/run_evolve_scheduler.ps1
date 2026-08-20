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
    & $pythonExe -m runtime_core.evolve_scheduler @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
