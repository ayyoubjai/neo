$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$envFile = Join-Path $root ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) {
            return
        }
        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ($key) {
            Set-Item -Path ("Env:" + $key) -Value $value
        }
    }
}

$settingsFile = $env:AGI_SETTINGS_PATH
if (-not $settingsFile) {
    $localSettingsFile = Join-Path $root "config/settings.local.json"
    $trackedSettingsFile = Join-Path $root "config/settings.json"
    $exampleSettingsFile = Join-Path $root "config/settings.example.json"
    if (Test-Path $localSettingsFile) {
        $settingsFile = $localSettingsFile
    } elseif (Test-Path $trackedSettingsFile) {
        $settingsFile = $trackedSettingsFile
    } else {
        $settingsFile = $exampleSettingsFile
    }
}

$settingsInterfaceModes = @()
$settingsInterfaceSenses = ""
$startAutonomyRuntime = $false
if (Test-Path $settingsFile) {
    try {
        $settings = Get-Content $settingsFile -Raw | ConvertFrom-Json
        if ($null -ne $settings.interface -and $null -ne $settings.interface.mode) {
            $rawMode = $settings.interface.mode
            if ($rawMode -is [System.Collections.IEnumerable] -and -not ($rawMode -is [string])) {
                $settingsInterfaceModes = @($rawMode | ForEach-Object { [string]$_ })
            } else {
                $settingsInterfaceModes = @([string]$rawMode)
            }
        }
        if ($null -ne $settings.interface -and $null -ne $settings.interface.senses) {
            $settingsInterfaceSenses = (($settings.interface.senses | ForEach-Object { [string]$_ }) -join ",")
        }
        if ($null -ne $settings.autonomy) {
            $autonomyEnabled = $false
            if ($null -ne $settings.autonomy.enabled) {
                $autonomyEnabled = [bool]$settings.autonomy.enabled
            } else {
                $autonomyEnabled = [bool]$settings.autonomy.epistemic_enabled -or [bool]$settings.autonomy.power_process_enabled
            }
            $startAutonomyRuntime = [bool]$settings.autonomy.start_with_run_all -and $autonomyEnabled -and (
                [bool]$settings.autonomy.epistemic_enabled -or [bool]$settings.autonomy.power_process_enabled
            )
        }
    } catch {
        $settingsInterfaceModes = @()
        $settingsInterfaceSenses = ""
        $startAutonomyRuntime = $false
    }
}

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
    Write-Error "Python not found. Set PYTHON env var or install Python."
    exit 1
}

$nodeExe = $null
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCmd) {
    $nodeExe = $nodeCmd.Source
}

$services = @(
    @{ Name = "storage"; Args = "-m storage.main" },
    @{ Name = "model"; Args = "-m model_server.main" },
    @{ Name = "tool"; Args = "-m tool_runtime.main" },
    @{ Name = "orchestrator"; Args = "-m orchestrator.main" },
    @{ Name = "evolve-scheduler"; Args = "-m runtime_core.evolve_scheduler" }
)

Push-Location $root
$procs = @()
foreach ($svc in $services) {
    Write-Host "[start] $($svc.Name)"
    $p = Start-Process -FilePath $pythonExe -ArgumentList $svc.Args -WorkingDirectory $root -PassThru -NoNewWindow
    $procs += $p
}

Start-Sleep -Milliseconds 500

if ($startAutonomyRuntime) {
    Write-Host "[start] autonomy-runtime"
    $procs += Start-Process -FilePath $pythonExe -ArgumentList "-m autonomy.runtime" -WorkingDirectory $root -PassThru -NoNewWindow
}

$interfaceProcs = @()

try {
    if (-not $settingsInterfaceModes -or $settingsInterfaceModes.Count -eq 0) {
        $settingsInterfaceModes = @("text")
    }

    $normalizedInterfaceModes = @(
        $settingsInterfaceModes |
        ForEach-Object { ([string]$_).Trim().ToLowerInvariant() } |
        Where-Object { $_ }
    )
    if (-not $normalizedInterfaceModes -or $normalizedInterfaceModes.Count -eq 0) {
        $normalizedInterfaceModes = @("text")
    }

    if ($normalizedInterfaceModes.Count -gt 1) {
        foreach ($mode in $normalizedInterfaceModes) {
            if ($mode -notin @("telegram", "whatsapp")) {
                Write-Error "Multiple interface modes currently support only 'telegram' and 'whatsapp'."
                exit 1
            }
        }

        $interfaceProcs = @()
        foreach ($mode in $normalizedInterfaceModes) {
            switch ($mode) {
                "telegram" {
                    Write-Host "[start] telegram"
                    $interfaceProcs += Start-Process -FilePath $pythonExe -ArgumentList "-m telegram_daemon.main" -WorkingDirectory $root -PassThru -NoNewWindow
                }
                "whatsapp" {
                    if (-not $nodeExe) {
                        Write-Error "WhatsApp mode requires Node.js 20+ and npm install."
                        exit 1
                    }
                    Write-Host "[start] whatsapp"
                    $interfaceProcs += Start-Process -FilePath $nodeExe -ArgumentList ".\src\whatsapp_daemon\main.mjs" -WorkingDirectory $root -PassThru -NoNewWindow
                }
            }
        }

        while ($true) {
            foreach ($proc in $interfaceProcs) {
                if ($proc.HasExited) {
                    if ($proc.ExitCode -ne 0) {
                        throw "Interface process exited with code $($proc.ExitCode)."
                    }
                    return
                }
            }
            Start-Sleep -Milliseconds 500
        }
    }

    $rawInterfaceMode = $settingsInterfaceModes[0]
    $interfaceMode = $normalizedInterfaceModes[0]

    switch ($interfaceMode) {
        "text" {
            Write-Host "[start] voice (text)"
            & $pythonExe -m voice_daemon.main --mode text
        }
        "audio" {
            Write-Host "[start] voice (audio)"
            & $pythonExe -m voice_daemon.main --mode audio
        }
        "vision" {
            Write-Host "[start] voice (vision)"
            & $pythonExe -m voice_daemon.main --mode vision
        }
        "local" {
            Write-Host "[start] voice (local)"
            if ($settingsInterfaceSenses) {
                & $pythonExe -m voice_daemon.main --mode local --senses $settingsInterfaceSenses
            } else {
                & $pythonExe -m voice_daemon.main --mode local
            }
        }
        "telegram" {
            Write-Host "[start] telegram"
            & $pythonExe -m telegram_daemon.main
        }
        "whatsapp" {
            if (-not $nodeExe) {
                Write-Error "WhatsApp mode requires Node.js 20+ and npm install."
                exit 1
            }
            Write-Host "[start] whatsapp"
            & $nodeExe .\src\whatsapp_daemon\main.mjs
        }
        default {
            Write-Error "Unknown interface mode '$rawInterfaceMode'. Use 'text', 'audio', 'vision', 'local', 'telegram', or 'whatsapp'."
            exit 1
        }
    }
} finally {
    if ($interfaceProcs) {
        foreach ($p in $interfaceProcs) {
            if ($p -and -not $p.HasExited) {
                try { $p.Kill() } catch {}
            }
        }
    }
    foreach ($p in $procs) {
        if ($p -and -not $p.HasExited) {
            try { $p.Kill() } catch {}
        }
    }
    Pop-Location
}
