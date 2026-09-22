param()

# start.ps1 — Start llama-server (router mode) and the assistant on Windows/PowerShell
# Behavior mirrors start.sh: when models.provider == 'llamacpp' this will start
# `llama-server --models-dir <dir> --host 127.0.0.1 --port 8080` if not already running.

Set-StrictMode -Version Latest
Set-Location $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'

function Read-Settings {
    $settingsFile = $env:AGI_SETTINGS_PATH
    if (-Not $settingsFile) {
        foreach ($candidate in @('settings.local.json', 'settings.json', 'settings.example.json')) {
            $path = Join-Path $PSScriptRoot "config\$candidate"
            if (Test-Path $path) { $settingsFile = $path; break }
        }
    }
    if (-Not $settingsFile -or -Not (Test-Path $settingsFile)) { return $null }
    try {
        return Get-Content $settingsFile -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

Write-Host "[start.ps1] Loading settings..."
$settings = Read-Settings
$provider = 'llamacpp'
if ($settings -and $settings.models -and $settings.models.provider) {
    $provider = $settings.models.provider
}

if ($provider -eq 'llamacpp') {
    $modelsDir = './models'
    if ($settings -and $settings.models -and $settings.models.models_dir) {
        $modelsDir = $settings.models.models_dir
    }

    # Generation must succeed: stale presets can violate memory policy.
    $modelsPreset = Join-Path -Path $PSScriptRoot -ChildPath 'config\models.ini'
    try {
        Write-Host "[start.ps1] Generating models preset -> $modelsPreset"
        $cfgDir = Join-Path -Path $PSScriptRoot -ChildPath 'config'
        if (-Not (Test-Path $cfgDir)) { New-Item -ItemType Directory -Path $cfgDir | Out-Null }
        & python "$PSScriptRoot/scripts/generate_models_ini.py" --models-dir "$modelsDir" --output "$modelsPreset" 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'Model preset generation failed' }
    } catch {
        Write-Error "[start.ps1] Could not generate models.ini: $_"
        exit 1
    }

    $healthUrl = 'http://127.0.0.1:8080/health'
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        Write-Host "[start.ps1] llama-server already responding at $healthUrl — skipping start."
    } catch {
        Write-Host "[start.ps1] Starting llama-server with models dir: $modelsDir"
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $serverExecutable = 'llama-server'
        if ($settings -and $settings.models -and
            $settings.models.PSObject.Properties['llama_server'] -and $settings.models.llama_server) {
            $serverExecutable = $settings.models.llama_server
        }
        if ($env:LLAMA_SERVER_BIN) { $serverExecutable = $env:LLAMA_SERVER_BIN }
        $psi.FileName = (Get-Command python).Source
        $psi.Arguments = "-m model_server.managed_router --server-bin `"$serverExecutable`" --models-preset `"$modelsPreset`" --host 127.0.0.1 --port 8080"
        $psi.RedirectStandardOutput = $false
        $psi.RedirectStandardError = $false
        $psi.UseShellExecute = $true
        $proc = [System.Diagnostics.Process]::Start($psi)

        Write-Host "[start.ps1] Waiting for llama-server to be ready (up to 60s)..."
        $ready = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 1
            try {
                Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null
                $ready = $true
                break
            } catch {
                if ($proc.HasExited) { Write-Error "llama-server process exited unexpectedly."; break }
            }
        }
        if (-Not $ready) { Write-Error "llama-server did not respond within 60 seconds."; exit 1 }
        Write-Host "[start.ps1] llama-server is ready → $healthUrl"
    }
} else {
    Write-Host "[start.ps1] Provider is '$provider' — expected 'llamacpp' to auto-start llama-server."
}

Write-Host "[start.ps1] Done. You can now start the assistant components as needed."
