"""SearXNG web search client."""

from urllib.parse import urlencode

import httpx

from .config import SEARXNG_URL


async def search(query: str, max_results: int = 10) -> str:
    """Search the web using SearXNG. Returns formatted results."""
    params = {"q": query, "format": "json", "categories": "general"}
    url = f"{SEARXNG_URL}/search?{urlencode(params)}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            return f"Search error: {e}"
        except Exception as e:
            return f"Error: {e}"

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
