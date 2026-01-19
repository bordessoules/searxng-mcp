"""Centralized configuration from environment variables."""

import os
from pathlib import Path

# Load .env file if it exists
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Server
SERVER_NAME = os.getenv("MCP_SERVER_NAME", "mcp-gateway")
SERVER_PORT = int(os.getenv("MCP_PORT", "8000"))

# SearXNG
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

# Summarization LLM
SUMMARIZE_ENABLED = os.getenv("SUMMARIZE_ENABLED", "false").lower() == "true"
SUMMARIZE_API_URL = os.getenv("SUMMARIZE_API_URL", "http://localhost:1234/v1")
SUMMARIZE_API_KEY = os.getenv("SUMMARIZE_API_KEY", "")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "qwen3-vl-4b")
SUMMARIZE_MAX_TOKENS = int(os.getenv("SUMMARIZE_MAX_OUTPUT_TOKENS", "1000"))
SUMMARIZE_TIMEOUT = int(os.getenv("SUMMARIZE_TIMEOUT", "120"))

# Playwright
PLAYWRIGHT_TOKEN = os.getenv("PLAYWRIGHT_MCP_TOKEN", "")
PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"

# Document cache
DOCUMENT_CACHE_DIR = os.getenv("DOCUMENT_CACHE_DIR", "")
