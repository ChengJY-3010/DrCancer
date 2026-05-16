$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

$ports = @(3000, 8000)

foreach ($port in $ports) {
  $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
  if (-not $connections) {
    continue
  }

  $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($pid in $pids) {
    try {
      Stop-Process -Id $pid -Force -ErrorAction Stop
      Write-Output "Stopped process $pid on port $port"
    } catch {
      Write-Output "Could not stop process $pid on port $port"
    }
  }
}

if (Test-Path ".next") {
  Remove-Item -Recurse -Force ".next"
  Write-Output "Cleared .next cache"
}

Write-Output "Starting frontend on http://127.0.0.1:3000"
Write-Output "Starting backend on http://127.0.0.1:8000"
npm run dev:all
