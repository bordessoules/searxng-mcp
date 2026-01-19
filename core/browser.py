"""Playwright browser handler for JS-heavy sites."""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import PLAYWRIGHT_HEADLESS, PLAYWRIGHT_TOKEN
from .summarizer import is_enabled, summarize, summarize_with_vision

MAX_CONTENT_LENGTH = 10000


async def fetch(url: str, prompt: str | None = None) -> str:
    """
    Fetch content using Chrome browser via Playwright MCP.

    Args:
        url: The URL to browse
        prompt: Optional summarization prompt

    Returns:
        Page content, optionally summarized with vision AI
    """
    result = await _browse(url)

    if "error" in result:
        return f"Browser error: {result['error']}"

    content = result.get("content", "")
    screenshot = result.get("screenshot", "")
    final_url = result.get("url", url)

    return await _format_result(final_url, content, screenshot, prompt)


async def _format_result(url: str, content: str, screenshot: str, prompt: str | None) -> str:
    """Format browser result with optional summarization."""
    if is_enabled():
        if screenshot:
            body = await summarize_with_vision(content, screenshot, prompt)
        elif prompt:
            body = await summarize(content, prompt)
        else:
            body = _truncate(content)
    else:
        body = _truncate(content)

    return f"**URL:** {url}\n\n{body}"


def _truncate(content: str) -> str:
    """Truncate content if too long."""
    if len(content) > MAX_CONTENT_LENGTH:
        return content[:MAX_CONTENT_LENGTH] + "\n\n...(truncated)"
    return content


async def _browse(url: str) -> dict:
    """Connect to Playwright MCP and browse URL."""
    server_params = _build_server_params()

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("browser_navigate", {"url": url})
                await asyncio.sleep(3)  # Wait for JS to render

                content = await _get_snapshot_text(session)
                screenshot = await _get_screenshot(session)

                return {"content": content, "screenshot": screenshot, "url": url}

    except Exception as e:
        return {"error": str(e)}


def _build_server_params() -> StdioServerParameters:
    """Build Playwright MCP server parameters from config."""
    env = {"PATH": os.environ.get("PATH", "")}
    args = ["@playwright/mcp@latest"]

    if PLAYWRIGHT_TOKEN:
        env["PLAYWRIGHT_MCP_EXTENSION_TOKEN"] = PLAYWRIGHT_TOKEN
        args.append("--extension")
    else:
        if PLAYWRIGHT_HEADLESS:
            args.append("--headless")
        args.extend(["--browser", "chrome"])

    return StdioServerParameters(command="npx", args=args, env=env)


async def _get_snapshot_text(session: ClientSession) -> str:
    """Extract text content from browser snapshot."""
    snapshot = await session.call_tool("browser_snapshot", {})
    if not snapshot or not snapshot.content:
        return ""
    return "".join(item.text for item in snapshot.content if hasattr(item, "text"))


async def _get_screenshot(session: ClientSession) -> str:
    """Capture screenshot if browser is visible."""
    if PLAYWRIGHT_HEADLESS and not PLAYWRIGHT_TOKEN:
        return ""

    try:
        result = await session.call_tool("browser_screenshot", {})
        if not result or not result.content:
            return ""
        for item in result.content:
            if hasattr(item, "data"):
                return item.data
    except Exception:
        pass
    return ""
