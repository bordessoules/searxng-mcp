"""Centralized configuration from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists
_env_file = Path(__file__).parent.parent / ".env"
load_dotenv(_env_file)

# =============================================================================
# User-configurable settings (via environment variables)
# =============================================================================

# Server
SERVER_PORT = int(os.getenv("MCP_PORT", "8000"))

# SearXNG (Docker recipe at docker/searxng/)
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

# Docling service (Docker recipe at docker/docling/)
DOCLING_URL = os.getenv("DOCLING_URL", "http://localhost:8001")

# Docling GPU service (CUDA-accelerated, uses docling-serve)
DOCLING_GPU_URL = os.getenv("DOCLING_GPU_URL", "http://localhost:8002")
USE_DOCLING_GPU = os.getenv("USE_DOCLING_GPU", "false").lower() == "true"

# Vision/Summarization LLM (external API - LM Studio, vLLM, OpenRouter, etc.)
SUMMARIZE_ENABLED = os.getenv("SUMMARIZE_ENABLED", "true").lower() == "true"
SUMMARIZE_API_URL = os.getenv("SUMMARIZE_API_URL", "http://localhost:1234/v1")
SUMMARIZE_API_KEY = os.getenv("SUMMARIZE_API_KEY", "")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "qwen3-vl-8b")
SUMMARIZE_TIMEOUT = int(os.getenv("SUMMARIZE_TIMEOUT", "120"))

# Playwright browser (extension mode by default)
PLAYWRIGHT_TOKEN = os.getenv("PLAYWRIGHT_MCP_TOKEN", "")
PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() == "true"

# =============================================================================
# Internal constants (sensible defaults, rarely need changing)
# =============================================================================

# Timeouts (seconds)
HTTP_TIMEOUT = 30.0
DOCUMENT_TIMEOUT = 300.0  # 5 min - VLM image descriptions take time
IMAGE_TIMEOUT = 60.0

# Browser settings
BROWSER_MAX_CONTENT_LENGTH = 10000
BROWSER_JS_RENDER_WAIT = 5.0
BROWSER_VIEWPORT_WIDTH = 1920
BROWSER_VIEWPORT_HEIGHT = 1080

# Content limits (characters)
DOCUMENT_MAX_LENGTH = int(os.getenv("DOCUMENT_MAX_LENGTH", "1000000"))  # 1MB default
SUMMARIZE_CONTENT_LIMIT = 100000
SUMMARIZE_VISION_CONTENT_LIMIT = 50000

# HTTP retry settings
HTTP_MAX_RETRIES = 2
