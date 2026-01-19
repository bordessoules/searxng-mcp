# MCP Gateway runner for Windows
param(
    [string]$Transport = "stdio",
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000
)

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
}

# Check if venv exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    uv venv
}

# Sync dependencies
Write-Host "Syncing dependencies..." -ForegroundColor Cyan
uv sync

# Load .env if exists
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

# Run gateway
Write-Host "Starting MCP Gateway (transport: $Transport)..." -ForegroundColor Green
uv run python gateway.py --transport $Transport --host $Host --port $Port
