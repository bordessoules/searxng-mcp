"""URL analysis and routing logic."""

from urllib.parse import urlparse
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'})
DOCUMENT_EXTENSIONS = frozenset({'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.xls', '.ppt'})

JS_HEAVY_DOMAINS = (
    "aliexpress", "amazon", "ebay", "taobao", "alibaba",
    "walmart", "target", "bestbuy", "newegg", "etsy",
    "facebook", "instagram", "twitter", "x.com", "tiktok",
    "linkedin", "reddit", "pinterest",
    "airbnb", "booking", "expedia", "tripadvisor",
    "youtube", "netflix", "spotify", "figma", "notion",
)


def _get_extension(url: str) -> str:
    """Extract file extension from URL."""
    path = urlparse(url).path.split("?")[0]
    return Path(path).suffix.lower()


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    return urlparse(url).netloc.lower()


def route(url: str) -> str:
    """
    Determine which handler to use for a URL.

    Returns: 'image', 'document', 'browser', or 'http'
    """
    ext = _get_extension(url)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"

    domain = _get_domain(url)
    if any(d in domain for d in JS_HEAVY_DOMAINS):
        return "browser"

    return "http"
