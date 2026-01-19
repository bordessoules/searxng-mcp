"""HTTP fetch handler for static web pages."""

import asyncio

import httpx

from .extractor import extract
from .summarizer import is_enabled, summarize

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# HTTP status codes that trigger browser fallback
FALLBACK_CODES = {403, 429}

# HTTP status codes that should be retried
RETRY_CODES = {500, 502, 503, 504}

MAX_RETRIES = 2


async def fetch(url: str, prompt: str | None = None, max_length: int = 50000) -> str:
    """
    Fetch and extract content from a URL via HTTP.
    Falls back to browser on errors or garbled content.
    """
    from . import browser  # Import once at top of function

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            response = await _fetch_with_retry(client, url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in FALLBACK_CODES:
                return await browser.fetch(url, prompt)
            return f"Fetch error: {e}"
        except httpx.HTTPError as e:
            return f"Fetch error: {e}"
        except Exception as e:
            return f"Error: {e}"

    content_type = response.headers.get("content-type", "")
    is_html = "text/html" in content_type

    if is_html:
        content, _ = extract(response.text)
        content = _clean(content)
    else:
        content = response.text

    # Fallback to browser if content is garbled or too short
    if _is_garbled(content) or (is_html and len(content.strip()) < 50):
        return await browser.fetch(url, prompt)

    if len(content) > max_length:
        content = content[:max_length] + "\n\n...(truncated)"

    if prompt and is_enabled():
        summary = await summarize(content, prompt)
        return f"## Summary\n\n{summary}\n\n---\n\n{content}"

    return content


async def _fetch_with_retry(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Fetch with retry on transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url, headers=HEADERS)
            if response.status_code in RETRY_CODES and attempt < MAX_RETRIES:
                await asyncio.sleep(1 * (attempt + 1))  # backoff
                continue
            return response
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            raise

    raise last_error or httpx.HTTPError("Max retries exceeded")


def _is_garbled(content: str) -> bool:
    """Detect garbled/binary content."""
    if not content or len(content) < 100:
        return False

    sample = content[:2000]
    # Count non-ASCII and replacement characters
    bad_chars = sum(1 for c in sample if ord(c) > 127 or c == '\ufffd')
    return bad_chars / len(sample) > 0.15


def _clean(text: str) -> str:
    """Clean whitespace from extracted content."""
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)
