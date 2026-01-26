"""
MCP Gateway Server

Simple interface for LLMs to access web content.
Tools: search(), get()
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
import core
from core.config import SERVER_PORT
from core.document import check_gpu_service
from core.logger import get_logger

log = get_logger("gateway")

mcp = FastMCP("mcp-gateway", host="0.0.0.0")

# ASGI app for uvicorn compatibility
app = mcp.sse_app()


async def startup_health_check() -> None:
    """Run health checks on startup and log warnings."""
    log.info("Running startup health checks...")

    gpu_status = await check_gpu_service()
    message = gpu_status["message"]

    if gpu_status["gpu_available"]:
        log.info(f"[OK] {message}")
    elif gpu_status["gpu_enabled"]:
        log.warning(f"[WARNING] {message}")
        log.warning("[WARNING] PDF parsing will fall back to SLOW local CPU mode!")
        log.warning("[WARNING] Start GPU service: cd docker/docling && docker-compose -f docker-compose.gpu.yml up -d")
    else:
        log.info(f"[INFO] {message}")


@mcp.tool()
async def search(query: str, max_results: int = 10) -> str:
    """
    Search the web and return results with titles, URLs, and snippets.

    Use this to find information and discover URLs.
    Then use 'get' to fetch the content from a URL.

    Args:
        query: Search query
        max_results: Max results to return (default: 10)

    Example: search("Python asyncio tutorial")
    """
    return await core.search(query, max_results)


@mcp.tool()
async def get(
    url: str,
    prompt: str | None = None,
    mode: str = "toc",
    chunk_id: int | None = None
) -> str:
    """
    Fetch content from any URL with smart routing and caching.

    ROUTING (automatic):
    - Images (.png, .jpg, .gif) → Vision AI description
    - Documents (.pdf, .docx) → Docling parser with VLM image descriptions
    - Web pages → Browser + VLM extraction → clean markdown

    RETRIEVAL MODES (for documents AND web pages):
    - mode="toc" (default): Table of contents with sections, token counts (~3KB)
    - mode="chunk" + chunk_id=N: Specific section content (~1-4KB per chunk)
    - mode="full": Complete content (WARNING: may be truncated by AI client)

    WORKFLOW:
    1. get(url) returns TOC with sections and chunk IDs
    2. get(url, mode="chunk", chunk_id=N) to read specific sections
    3. Navigate using prev_chunk/next_chunk in response

    Args:
        url: Any URL (web page, PDF, document, image)
        prompt: AI instruction for images only (e.g., "describe this diagram")
        mode: "toc" (default), "chunk", or "full"
        chunk_id: Section index (required when mode="chunk")

    Returns:
        - mode="toc": JSON with source_url, sections [{heading, preview, tokens, chunk_id}]
        - mode="chunk": JSON with source_url, heading, text, prev/next navigation
        - mode="full": Raw markdown (may be large, risk of truncation)

    Citation: All structured responses include source_url for proper attribution.

    Examples:
        get("https://arxiv.org/pdf/1706.03762.pdf")  # PDF → TOC
        get("https://arxiv.org/pdf/1706.03762.pdf", mode="chunk", chunk_id=7)
        get("https://en.wikipedia.org/wiki/Transformer")  # Web page → TOC
        get("https://en.wikipedia.org/wiki/Transformer", mode="chunk", chunk_id=3)
        get("https://example.com/photo.jpg", prompt="describe")  # Image
    """
    return await core.get(url, prompt, mode=mode, chunk_id=chunk_id)


def run_stdio() -> None:
    mcp.run(transport="stdio")


def run_sse(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    # Run health check before starting
    asyncio.run(startup_health_check())

    print(f"\nStarting MCP Gateway (transport: sse)...")
    uvicorn.run(mcp.sse_app(), host=host, port=port or SERVER_PORT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Gateway")
    parser.add_argument("-t", "--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=SERVER_PORT)

    args = parser.parse_args()

    if args.transport == "stdio":
        run_stdio()
    else:
        run_sse(args.host, args.port)
