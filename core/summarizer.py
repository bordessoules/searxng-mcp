"""LLM summarization via OpenAI-compatible API."""

import httpx

from .config import (
    SUMMARIZE_API_KEY,
    SUMMARIZE_API_URL,
    SUMMARIZE_ENABLED,
    SUMMARIZE_MAX_TOKENS,
    SUMMARIZE_MODEL,
    SUMMARIZE_TIMEOUT,
)


def is_enabled() -> bool:
    return SUMMARIZE_ENABLED


async def summarize(content: str, prompt: str | None = None) -> str:
    """Summarize text content using LLM."""
    if not content:
        return ""

    system = prompt or "Summarize the content concisely. Focus on key facts."

    return await _call_api([
        {"role": "system", "content": system},
        {"role": "user", "content": content[:100000]},  # Truncate input
    ])


async def summarize_with_vision(
    content: str,
    screenshot_b64: str,
    prompt: str | None = None
) -> str:
    """Summarize with both text and screenshot."""
    if not content and not screenshot_b64:
        return ""

    system = prompt or "Analyze this page. Extract key information concisely."

    user_content = []
    if screenshot_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
        })
    if content:
        user_content.append({"type": "text", "text": content[:50000]})

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
        "temperature": 0.3,
        "max_tokens": SUMMARIZE_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=SUMMARIZE_TIMEOUT) as client:
            resp = await client.post(
                f"{SUMMARIZE_API_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Summarization error: {e}"
