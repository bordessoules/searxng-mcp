#!/bin/bash
# MCP Gateway runner for Linux/Mac

TRANSPORT="${1:-stdio}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Load .env if exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run gateway
echo "Starting MCP Gateway (transport: $TRANSPORT)..."
uv run python gateway.py --transport "$TRANSPORT" --host "$HOST" --port "$PORT"
