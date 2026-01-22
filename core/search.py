"""SearXNG web search client."""

from urllib.parse import urlencode

import httpx

from .config import HTTP_TIMEOUT, SEARXNG_URL
from .logger import get_logger

log = get_logger("search")


async def search(query: str, max_results: int = 10) -> str:
    """Search the web using SearXNG. Returns formatted results."""
    params = {"q": query, "format": "json", "categories": "general"}
    url = f"{SEARXNG_URL}/search?{urlencode(params)}"

    log.debug(f"Searching for: {query}")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            log.error(f"Search API returned HTTP {e.response.status_code}")
            return f"Search error: HTTP {e.response.status_code}"
        except httpx.TimeoutException:
            log.error(f"Search request timed out for query: {query}")
            return "Search error: Request timed out"
        except httpx.RequestError as e:
            log.error(f"Search request failed: {e}")
            return f"Search error: {e}"
        except ValueError as e:
            log.error(f"Failed to parse search response as JSON: {e}")
            return "Search error: Invalid response format"

    results = data.get("results", [])[:max_results]
    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        output.append(f"## {i}. {title}\n**URL:** {url}\n{snippet}\n")

    return "\n".join(output)
