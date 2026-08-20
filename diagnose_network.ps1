# Diagnose Network Connection Issues

Write-Host "🔍 Checking network services..." -ForegroundColor Cyan

# Check Ollama (default port 11434)
Write-Host "`n[1] Checking Ollama service..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/version" -ErrorAction Stop -TimeoutSec 2
    Write-Host "✅ Ollama is running on port 11434" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama NOT responding on port 11434" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Check if Ollama process exists
Write-Host "`n[2] Checking if Ollama process is running..." -ForegroundColor Yellow
$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "✅ Ollama process found (PID: $($ollama.Id))" -ForegroundColor Green
} else {
    Write-Host "❌ Ollama process NOT found" -ForegroundColor Red
    Write-Host "   → Run 'ollama serve' in a terminal" -ForegroundColor Yellow
}

# Check Neo4j (default port 7687)
Write-Host "`n[3] Checking Neo4j service..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:7474" -ErrorAction Stop -TimeoutSec 2
    Write-Host "✅ Neo4j is running on port 7474" -ForegroundColor Green
} catch {
    Write-Host "❌ Neo4j NOT responding on port 7474" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Check if docker container is running
Write-Host "`n[4] Checking Docker containers..." -ForegroundColor Yellow
try {
    $containers = docker ps --filter "name=neo4j" --format "{{.Names}}"
    if ($containers) {
        Write-Host "✅ Neo4j Docker container is running: $containers" -ForegroundColor Green
    } else {
        Write-Host "❌ Neo4j Docker container NOT running" -ForegroundColor Red
        Write-Host "   → Run 'docker-compose -f docker-compose.neo4j.yml up -d'" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Docker not available or command failed" -ForegroundColor Yellow
}

# Check listening ports
Write-Host "`n[5] Checking listening ports..." -ForegroundColor Yellow
$ports = @(11434, 7474, 7687)
foreach ($port in $ports) {
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        Write-Host "✅ Port $port is listening" -ForegroundColor Green
    } else {
        Write-Host "❌ Port $port is NOT listening" -ForegroundColor Red
    }
}

# Summary
Write-Host "`n" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "1. Start Ollama: ollama serve" -ForegroundColor Yellow
Write-Host "2. Start Neo4j: docker-compose -f docker-compose.neo4j.yml up -d" -ForegroundColor Yellow
Write-Host "3. Verify with: python epistemic_diagnostics.py" -ForegroundColor Yellow
Write-Host "4. Try again: python -m autonomy.runtime" -ForegroundColor Yellow
