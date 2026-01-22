"""LLM summarization via OpenAI-compatible API."""

import httpx

from .logger import get_logger
from .config import (
    SUMMARIZE_API_KEY,
    SUMMARIZE_API_URL,
    SUMMARIZE_CONTENT_LIMIT,
    SUMMARIZE_ENABLED,
    SUMMARIZE_MODEL,
    SUMMARIZE_TIMEOUT,
    SUMMARIZE_VISION_CONTENT_LIMIT,
)

log = get_logger("summarizer")


def is_enabled() -> bool:
    return SUMMARIZE_ENABLED


async def summarize(content: str, prompt: str | None = None) -> str:
    """Summarize text content using LLM."""
    if not content:
        return ""

    system = prompt or "Summarize the content concisely. Focus on key facts."

    return await _call_api([
        {"role": "system", "content": system},
        {"role": "user", "content": content[:SUMMARIZE_CONTENT_LIMIT]},
    ])


async def summarize_with_vision(
    content: str,
    screenshot_b64: str,
    prompt: str | None = None,
    url: str | None = None
) -> str:
    """Summarize with screenshot, text content, and URL context."""
    if not content and not screenshot_b64:
        return ""

    # Always provide context about what we're analyzing
    base_context = f"""Analyzing: {url or 'webpage'}
You receive a screenshot and the page's text content."""

    if prompt:
        # User prompt becomes the task
        system = f"{base_context}\n\nTask: {prompt}"
    else:
        # Default extraction task
        system = f"""{base_context}

Extract key information as clean, readable markdown:
- Main content and purpose
- Prices, specs, details (if applicable)
- Important facts and data

Ignore navigation, ads, and boilerplate."""

    user_content = []
    if screenshot_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
        })
    if content:
        user_content.append({"type": "text", "text": content[:SUMMARIZE_VISION_CONTENT_LIMIT]})

    return await _call_api([
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ])


async def _call_api(messages: list) -> str:
    """Make API call to LLM."""
    headers = {"Content-Type": "application/json"}
    if SUMMARIZE_API_KEY:
        headers["Authorization"] = f"Bearer {SUMMARIZE_API_KEY}"

    payload = {
        "model": SUMMARIZE_MODEL,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=float(SUMMARIZE_TIMEOUT)) as client:
            resp = await client.post(
                f"{SUMMARIZE_API_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            # Safely extract response content with validation
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                log.error("Invalid API response: missing 'choices' array")
                return "Summarization error: Invalid API response format"

            message = choices[0].get("message") if choices else None
            if not message or not isinstance(message, dict):
                log.error("Invalid API response: missing 'message' object")
                return "Summarization error: Invalid API response format"

            content = message.get("content", "")
            if not content:
                log.warning("API returned empty content")
            return content

    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error from summarization API: {e.response.status_code}")
        return f"Summarization error: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        log.error(f"Request error calling summarization API: {e}")
        return f"Summarization error: {e}"
    except (ValueError, KeyError, TypeError) as e:
        log.error(f"Error parsing API response: {e}")
        return f"Summarization error: Invalid response - {e}"
