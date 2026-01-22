# MCP Gateway

A modular MCP (Model Context Protocol) server that provides LLMs with web access capabilities. Exposes two simple tools: `search()` and `get()`.

## Features

- **search(query)** - Web search via self-hosted SearXNG (meta search engine)
- **get(url)** - Smart content fetching with automatic routing:
  - Images (.png, .jpg, .webp) → Vision AI description
  - Documents (PDF, DOCX, XLSX) → Docling parsing
  - Web pages → Playwright browser + Vision AI (consistent quality for all sites)

## Architecture

```
┌─────────────────┐     SSE/HTTP      ┌──────────────────┐
│   LLM Client    │ ◄───────────────► │   MCP Gateway    │
│   (Claude,      │     Port 8000     │   (FastMCP)      │
│    etc.)        │                   └────────┬─────────┘
└─────────────────┘                            │
                                               ▼
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              ┌─────▼─────┐           ┌────────▼────────┐        ┌───────▼───────┐
              │  SearXNG  │           │   LM Studio     │        │   Playwright  │
              │  :8080    │           │   :1234         │        │   (Chrome)    │
              │  (search) │           │  (summarize/    │        │  (all web     │
              └───────────┘           │   vision)       │        │   pages)      │
                                      └─────────────────┘        └───────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for SearXNG)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Node.js (for Playwright MCP via npx)
- Optional: LM Studio for summarization/vision

### Setup

1. **Clone and install dependencies**
   ```bash
   git clone <repo-url>
   cd searxng-mcp
   uv sync  # or: pip install -e .
   ```

2. **Start SearXNG** (search engine)
   ```bash
   cd docker/searxng && docker compose up -d
   ```

3. **Start Docling** (document parser)
   ```bash
   # CPU version (simpler, no GPU required)
   cd docker/docling && docker compose up -d

   # OR GPU version with CUDA + VLM picture descriptions
   cd docker/docling && docker compose -f docker-compose.gpu.yml up -d
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run the gateway**
   ```bash
   # SSE mode (for network access)
   python gateway.py -t sse -p 8000

   # Or stdio mode (for local CLI)
   python gateway.py
   ```

## Configuration

Environment variables in `.env`:

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_PORT` | Gateway port | `8000` |
| `SEARXNG_URL` | SearXNG instance URL | `http://localhost:8080` |

### Docling Document Parser

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCLING_URL` | CPU Docling service URL | `http://localhost:8001` |
| `DOCLING_GPU_URL` | GPU Docling service URL (CUDA) | `http://localhost:8002` |
| `USE_DOCLING_GPU` | Enable GPU service | `false` |
| `DOCUMENT_MAX_LENGTH` | Max document chars returned | `1000000` |

### Vision/Summarization LLM

| Variable | Description | Default |
|----------|-------------|---------|
| `SUMMARIZE_ENABLED` | Enable LLM summarization | `false` |
| `SUMMARIZE_API_URL` | OpenAI-compatible API URL | `http://localhost:1234/v1` |
| `SUMMARIZE_MODEL` | Model for summarization/vision | `qwen3-vl-4b` |
| `SUMMARIZE_TIMEOUT` | API timeout (seconds) | `120` |

### Browser

| Variable | Description | Default |
|----------|-------------|---------|
| `PLAYWRIGHT_MCP_TOKEN` | Chrome extension token | - |
| `PLAYWRIGHT_HEADLESS` | Run browser headless | `true` |

## Usage

### From an MCP Client

```python
# Search the web
results = await search("Python asyncio tutorial", max_results=10)

# Get content from URL (uses Playwright + Vision AI)
content = await get("https://docs.python.org/3/")

# Get with summarization prompt
content = await get("https://amazon.com/dp/B123", "extract price and specs")

# Describe an image
description = await get("https://example.com/photo.jpg", "describe what you see")

# Parse a PDF
content = await get("https://arxiv.org/pdf/1706.03762.pdf")
```

### Docker Deployment

Full stack with Caddy reverse proxy:
```bash
docker compose up -d
```

This starts:
- SearXNG on port 8080
- MCP Gateway on port 8000 (via Caddy)

## Project Structure

```
searxng-mcp/
├── gateway.py          # Main entry point, tool definitions
├── core/
│   ├── __init__.py     # Exports search() and get()
│   ├── router.py       # URL routing logic
│   ├── search.py       # SearXNG integration
│   ├── browser.py      # Playwright integration
│   ├── image.py        # Vision AI handling
│   ├── document.py     # Docling PDF/DOCX parsing
│   ├── summarizer.py   # LLM summarization
│   ├── config.py       # Environment configuration
│   └── logger.py       # Centralized logging
├── scripts/
│   └── benchmark_image.py  # Performance benchmarks
├── searxng/            # SearXNG configuration
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## License

MIT
