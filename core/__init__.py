"""
MCP Gateway Core - Simple interface, smart routing.

Exports:
    search(query) - Web search via SearXNG
    get(url) - Get content from any URL (auto-routes)
    route(url) - Get routing decision for URL
"""

from .router import route
from .search import search as _search
from . import http, browser, document, image

_HANDLERS = {
    "image": image.fetch,
    "document": document.fetch,
    "browser": browser.fetch,
    "http": http.fetch,
}


async def search(query: str, max_results: int = 10) -> str:
    """Search the web, return URLs + snippets."""
    return await _search(query, max_results)


async def get(url: str, prompt: str | None = None) -> str:
    """
    Get content from any URL. Auto-routes based on content type.

    - Images (.png, .jpg) -> Vision model (Qwen3-VL)
    - Documents (.pdf, .docx) -> Docling parser
    - JS-heavy sites (amazon, etc.) -> Chrome browser
    - Everything else -> Fast HTTP fetch
    """
    handler = _HANDLERS[route(url)]
    return await handler(url, prompt)
