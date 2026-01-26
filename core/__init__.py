"""
MCP Gateway Core - Simple interface, smart routing.

Exports:
    search(query) - Web search via SearXNG
    get(url) - Get content from any URL (auto-routes)
    route(url) - Get routing decision for URL

Three-way routing:
    - Images → Vision AI description
    - Documents (.pdf, .docx) → Docling with VLM image descriptions
    - Web pages → Browser + VLM extraction → clean markdown

All document types support TOC/chunk navigation for efficient retrieval.
VLM model is configurable via SUMMARIZE_MODEL env var.
"""

from .router import route
from .search import search as _search
from . import document, image, webpage


async def search(query: str, max_results: int = 10) -> str:
    """Search the web, return URLs + snippets."""
    return await _search(query, max_results)


async def get(
    url: str,
    prompt: str | None = None,
    mode: str = "toc",
    chunk_id: int | None = None
) -> str:
    """
    Get content from any URL. Auto-routes based on content type.

    Three-way routing:
    - Images (.png, .jpg, .gif) → Vision AI description
    - Documents (.pdf, .docx, .xlsx) → Docling with TOC/chunk navigation
    - Web pages (everything else) → Browser + VLM extraction pipeline
      - Renders JavaScript
      - Screenshot + text → VLM → clean markdown
      - Then local chunking for TOC/chunk navigation

    For all non-image content, supports three retrieval modes:
    - mode="toc" (default): Table of contents with sections (~3KB)
    - mode="chunk": Specific chunk by ID with navigation (~1-4KB)
    - mode="full": Complete document (WARNING: may be truncated)

    All structured responses (toc/chunk) include source_url for citation.

    Args:
        url: URL to fetch
        prompt: Optional prompt for AI processing (used for images)
        mode: Retrieval mode ("toc", "chunk", "full")
        chunk_id: Chunk index when mode="chunk"
    """
    handler = route(url)

    if handler == "image":
        return await image.fetch(url, prompt)

    if handler == "document":
        return await document.fetch(url, prompt, mode=mode, chunk_id=chunk_id)

    # handler == "webpage"
    return await webpage.fetch(url, prompt, mode=mode, chunk_id=chunk_id)
