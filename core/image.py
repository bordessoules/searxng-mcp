"""Image handler using vision LLM."""

import base64

import httpx

from .config import IMAGE_TIMEOUT
from .logger import get_logger
from .summarizer import is_enabled, summarize_with_vision

log = get_logger("image")

DEFAULT_PROMPT = "Describe this image in detail. Extract any visible text."

IMAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "cross-site",
}


async def fetch(url: str, prompt: str | None = None) -> str:
    """Analyze image using vision model."""
    if not is_enabled():
        return "Error: Vision model not enabled. Set SUMMARIZE_ENABLED=true in .env"

    log.debug(f"Fetching image: {url}")

    try:
        async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers=IMAGE_HEADERS)
            resp.raise_for_status()
            img_bytes = resp.content
            log.debug(f"Downloaded image: {len(img_bytes)} bytes")
    except httpx.HTTPStatusError as e:
        log.error(f"Image download failed: HTTP {e.response.status_code}")
        return f"Image download error: HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        log.error(f"Image download timed out: {url}")
        return "Image download error: Request timed out"
    except httpx.RequestError as e:
        log.error(f"Image download error: {e}")
        return f"Image download error: {e}"

    img_b64 = base64.b64encode(img_bytes).decode()
    return await summarize_with_vision("", img_b64, prompt or DEFAULT_PROMPT)
