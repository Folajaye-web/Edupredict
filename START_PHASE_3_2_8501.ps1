$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host 'EduPredict Phase 3.2 - clean launcher' -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

# Stop only Streamlit processes that are serving EduPredict on port 8501.
$listeners = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listeners) {
    $pid = $conn.OwningProcess
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pid" -ErrorAction SilentlyContinue
    if ($proc -and ($proc.CommandLine -match 'streamlit|streamlit_app\.py|EduPredict')) {
        Write-Host "Stopping stale EduPredict process PID $pid ..." -ForegroundColor Yellow
        Stop-Process -Id $pid -Force
    } else {
        Write-Warning "Port 8501 is occupied by PID $pid, but it does not look like EduPredict. Close it manually before continuing."
        exit 1
    }
}

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Write-Host 'Starting the current Phase 3.2 application on port 8501 ...' -ForegroundColor Green
& $python -m streamlit run (Join-Path $ProjectRoot 'app\streamlit_app.py') --server.port 8501
