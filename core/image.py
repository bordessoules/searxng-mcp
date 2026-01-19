"""Image handler using vision LLM."""

import base64

import httpx

from .http import USER_AGENT
from .summarizer import is_enabled, summarize_with_vision

DEFAULT_PROMPT = "Describe this image in detail. Extract any visible text."

IMAGE_HEADERS = {
    "User-Agent": USER_AGENT,
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

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=IMAGE_HEADERS)
            resp.raise_for_status()
            img_bytes = resp.content
    except httpx.HTTPError as e:
        return f"Image download error: {e}"
    except Exception as e:
        return f"Image error: {e}"

    img_b64 = base64.b64encode(img_bytes).decode()
    return await summarize_with_vision("", img_b64, prompt or DEFAULT_PROMPT)
