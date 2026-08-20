Write-Host "Starting Neo4j container..."
docker compose -f c:\AGI\docker-compose.neo4j.yml up -d

Write-Host "Waiting 15 seconds for Neo4j to boot..."
Start-Sleep -Seconds 15

Write-Host "Verifying Neo4j connection..."
$response = Invoke-WebRequest -Uri "http://localhost:7474/" -ErrorAction SilentlyContinue
if ($response.StatusCode -eq 200) { 
    Write-Host "Neo4j is ready!" -ForegroundColor Green 
} else { 
    Write-Host "Neo4j not ready" -ForegroundColor Red 
}