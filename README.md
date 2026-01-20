# MCP Gateway

A modular MCP (Model Context Protocol) server that provides LLMs with web access capabilities. Exposes two simple tools: `search()` and `get()`.

## Features

- **search(query)** - Web search via self-hosted SearXNG (meta search engine)
- **get(url)** - Smart content fetching with automatic routing:
  - Images (.png, .jpg, .webp) → Vision AI description
  - Documents (PDF, DOCX, XLSX) → Docling parsing
  - JS-heavy sites (e-commerce, SPAs) → Playwright browser
  - Static pages → HTTP fetch + trafilatura extraction

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
              │  (search) │           │  (summarize/    │        │  (JS sites)   │
              └───────────┘           │   vision)       │        └───────────────┘
                                      └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for SearXNG)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
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
   docker compose up -d searxng
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Run the gateway**
   ```bash
   # SSE mode (for network access)
   python gateway.py -t sse -p 8000

   # Or stdio mode (for local CLI)
   python gateway.py
   ```

## Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `SEARXNG_URL` | SearXNG instance URL | `http://localhost:8080` |
| `MCP_PORT` | Gateway port | `8000` |
| `SUMMARIZE_ENABLED` | Enable LLM summarization | `false` |
| `SUMMARIZE_API_URL` | OpenAI-compatible API URL | `http://localhost:1234/v1` |
| `SUMMARIZE_MODEL` | Model for summarization | - |
| `PLAYWRIGHT_MCP_TOKEN` | Chrome extension token | - |

## Usage

### From an MCP Client

```python
# Search the web
results = await search("Python asyncio tutorial", max_results=10)

# Get content from URL
content = await get("https://docs.python.org/3/")

# Get with summarization prompt
content = await get("https://amazon.com/dp/B123", "extract price and specs")

# Describe an image
description = await get("https://example.com/photo.jpg", "describe what you see")
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
│   ├── http.py         # HTTP fetch with retry/fallback
│   ├── browser.py      # Playwright integration
│   ├── image.py        # Vision AI handling
│   ├── document.py     # Docling PDF/DOCX parsing
│   ├── extractor.py    # HTML content extraction
│   ├── summarizer.py   # LLM summarization
│   └── config.py       # Environment configuration
├── searxng/            # SearXNG configuration
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## License

MIT
