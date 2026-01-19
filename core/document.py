"""Document parsing handler using Docling."""

import hashlib
import logging
import os
import tempfile
import warnings
from pathlib import Path

import httpx

from .config import DOCUMENT_CACHE_DIR

# Suppress noisy logging from dependencies
for _name in ("docling", "rapidocr", "RapidOCR", "PIL"):
    logging.getLogger(_name).setLevel(logging.ERROR)
os.environ["RAPIDOCR_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

_converter = None
_cache_dir = None


def _get_converter():
    """Get or create Docling converter (lazy-loaded)."""
    global _converter
    if _converter is None:
        try:
            from docling.document_converter import DocumentConverter
            _converter = DocumentConverter()
        except ImportError:
            return None
    return _converter


def _get_cache_dir() -> Path:
    """Get or create cache directory."""
    global _cache_dir
    if _cache_dir is None:
        base = DOCUMENT_CACHE_DIR or tempfile.gettempdir()
        _cache_dir = Path(base) / "docling_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


async def fetch(url: str, prompt: str | None = None, max_length: int = 100000) -> str:
    """
    Parse a document from URL using Docling.

    Args:
        url: URL to document (PDF, DOCX, etc.)
        prompt: Ignored (documents don't need summarization prompt)
        max_length: Max characters to return

    Returns:
        Parsed document as markdown
    """
    converter = _get_converter()
    if converter is None:
        return "Error: Docling not installed. Run: pip install docling"

    try:
        file_path = await _download(url)
        result = converter.convert(str(file_path))
        content = result.document.export_to_markdown()

        if len(content) > max_length:
            content = content[:max_length] + "\n\n...(truncated)"

        return content

    except Exception as e:
        return f"Document error: {e}"


async def _download(url: str) -> Path:
    """Download file to cache."""
    url_path = url.split("?")[0]
    suffix = Path(url_path).suffix or ".pdf"
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    cache_path = _get_cache_dir() / f"{url_hash}{suffix}"

    if cache_path.exists():
        return cache_path

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        cache_path.write_bytes(response.content)

    return cache_path
