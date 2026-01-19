"""Content extraction from HTML using trafilatura/readability."""

import trafilatura
from markdownify import markdownify
from readability import Document

EXTRACTORS = ["trafilatura", "readability", "raw"]


def extract(html: str, mode: str = "auto") -> tuple[str, str]:
    """
    Extract content from HTML.

    Args:
        html: Raw HTML content
        mode: "auto", "trafilatura", "readability", or "raw"

    Returns:
        Tuple of (content, method_used)
    """
    if mode == "raw":
        return _raw(html), "raw"

    if mode in ("trafilatura", "readability"):
        result = _try_extract(html, mode)
        if result:
            return result, mode
        return _raw(html), "raw"

    # Auto mode: try each extractor in order
    for method in EXTRACTORS:
        result = _try_extract(html, method)
        if result:
            return result, method

    return _raw(html), "raw"


def _try_extract(html: str, method: str) -> str | None:
    """Try a specific extraction method."""
    if method == "trafilatura":
        return _trafilatura(html)
    if method == "readability":
        return _readability(html)
    if method == "raw":
        return _raw(html)
    return None


def _trafilatura(html: str) -> str | None:
    try:
        return trafilatura.extract(
            html,
            favor_recall=True,
            include_links=True,
            include_tables=True,
            output_format="markdown",
        )
    except Exception:
        return None


def _readability(html: str) -> str | None:
    try:
        doc = Document(html)
        content = doc.summary()
        return markdownify(content, strip=["script", "style"]) if content else None
    except Exception:
        return None


def _raw(html: str) -> str:
    return markdownify(html, strip=["script", "style"])
