"""
MCP Gateway Server

Simple interface for LLMs to access web content.
Tools: search(), get()
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
import core
from core.config import SERVER_PORT

mcp = FastMCP("mcp-gateway", host="0.0.0.0")

# ASGI app for uvicorn compatibility
app = mcp.sse_app()


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
async def get(url: str, prompt: str | None = None) -> str:
    """
    Get content from any URL. Auto-detects the best method.

    - Images (.png, .jpg) → Vision AI description (~5-7s)
    - Web pages → extracted text (~1-2s)
    - PDFs/DOCX → parsed with Docling (~30s first time)
    - E-commerce sites → Chrome browser + AI (~5-10s)

    Args:
        url: Any URL (web page, PDF, document, image)
        prompt: Optional instruction for AI (e.g., "extract price and specs")

    Example: get("https://docs.python.org/3/")
    Example: get("https://arxiv.org/pdf/1706.03762.pdf")
    Example: get("https://amazon.com/dp/B123", "get product details")
    Example: get("https://example.com/photo.jpg", "describe what you see")
    """
    return await core.get(url, prompt)


def run_stdio():
    mcp.run(transport="stdio")


def run_sse(host: str = "0.0.0.0", port: int = None):
    import uvicorn
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
