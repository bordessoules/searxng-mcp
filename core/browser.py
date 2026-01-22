"""
Browser handler using Playwright MCP with Chrome extension.

Connects to your real Chrome browser via the Playwright extension,
giving you access to all your extensions, logins, and cookies.
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import (
    BROWSER_JS_RENDER_WAIT,
    BROWSER_MAX_CONTENT_LENGTH,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
    PLAYWRIGHT_HEADLESS,
    PLAYWRIGHT_TOKEN,
)
from .logger import get_logger
from .summarizer import is_enabled, summarize, summarize_with_vision

log = get_logger("browser")


async def fetch(url: str, prompt: str | None = None) -> str:
    """
    Fetch content using Chrome browser via Playwright MCP extension.

    Uses your real Chrome with all extensions (adblocker, cookie consent, etc.)
    and existing logins/cookies.

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


async def _browse(url: str) -> dict:
    """Connect to Playwright MCP and browse URL using Chrome extension."""
    server_params = _build_server_params()

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Set viewport for consistent screenshots
                try:
                    await session.call_tool("browser_resize", {
                        "width": BROWSER_VIEWPORT_WIDTH,
                        "height": BROWSER_VIEWPORT_HEIGHT
                    })
                except Exception as e:
                    log.debug(f"Browser resize failed (non-critical): {e}")

                # Navigate to URL
                await session.call_tool("browser_navigate", {"url": url})

                # Wait for JS to render
                await asyncio.sleep(BROWSER_JS_RENDER_WAIT)

                # Extract content and screenshot
                content = await _get_snapshot_text(session)
                screenshot = await _get_screenshot(session)

                return {"content": content, "screenshot": screenshot, "url": url}

    except FileNotFoundError:
        log.error("npx not found. Please install Node.js: https://nodejs.org/")
        return {"error": "Node.js/npx not installed"}
    except Exception as e:
        log.error(f"Browser navigation failed for {url}: {e}")
        return {"error": str(e)}


def _build_server_params() -> StdioServerParameters:
    """Build Playwright MCP server parameters."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
    }

    # Use extension mode with token to connect to real Chrome
    if PLAYWRIGHT_TOKEN:
        env["PLAYWRIGHT_MCP_EXTENSION_TOKEN"] = PLAYWRIGHT_TOKEN
        args = ["@playwright/mcp@latest", "--extension"]
    elif PLAYWRIGHT_HEADLESS:
        # Fallback to headless if no token and headless enabled
        args = ["@playwright/mcp@latest", "--headless", "--browser", "chrome"]
    else:
        # Default: try extension mode without token
        args = ["@playwright/mcp@latest", "--extension"]

    return StdioServerParameters(command="npx", args=args, env=env)


async def _get_snapshot_text(session: ClientSession) -> str:
    """Extract text content from browser snapshot."""
    try:
        snapshot = await session.call_tool("browser_snapshot", {})
        if not snapshot or not snapshot.content:
            return ""
        return "".join(item.text for item in snapshot.content if hasattr(item, "text"))
    except Exception as e:
        log.debug(f"Snapshot failed: {e}")
        return ""


async def _get_screenshot(session: ClientSession) -> str:
    """Capture screenshot for vision AI."""
    try:
        result = await session.call_tool("browser_screenshot", {"fullPage": True})
        if not result or not result.content:
            result = await session.call_tool("browser_screenshot", {})
        if not result or not result.content:
            return ""
        for item in result.content:
            if hasattr(item, "data"):
                return item.data
    except Exception as e:
        log.debug(f"Screenshot failed: {e}")
    return ""


async def _format_result(url: str, content: str, screenshot: str, prompt: str | None) -> str:
    """Format browser result with optional vision summarization."""
    if is_enabled() and screenshot:
        body = await summarize_with_vision(content, screenshot, prompt, url=url)
    elif is_enabled() and prompt:
        body = await summarize(content, prompt)
    else:
        body = _truncate(content)

    return f"**URL:** {url}\n\n{body}"


def _truncate(content: str) -> str:
    """Truncate content if too long."""
    if len(content) > BROWSER_MAX_CONTENT_LENGTH:
        return content[:BROWSER_MAX_CONTENT_LENGTH] + "\n\n...(truncated)"
    return content
