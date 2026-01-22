"""Document parsing handler - calls Docling service or falls back to local."""

import hashlib
import tempfile
from pathlib import Path

import httpx

from .config import DOCLING_URL, DOCLING_GPU_URL, USE_DOCLING_GPU, DOCUMENT_TIMEOUT, DOCUMENT_MAX_LENGTH
from .logger import get_logger

log = get_logger("document")


async def fetch(url: str, prompt: str | None = None, max_length: int = DOCUMENT_MAX_LENGTH) -> str:
    """
    Parse a document from URL using Docling service.

    Tries the Docker Docling service first, falls back to local if unavailable.

    Args:
        url: URL to document (PDF, DOCX, etc.)
        prompt: Ignored (documents don't need summarization prompt)
        max_length: Max characters to return

    Returns:
        Parsed document as markdown
    """
    # Try Docling service first
    content = await _fetch_from_service(url)

    if content is None:
        # Fallback to local Docling if service unavailable
        content = await _fetch_local(url)

    if content is None:
        return "Error: Docling service unavailable and local Docling not installed."

    log.info(f"Content length: {len(content)}, max_length: {max_length}")
    if len(content) > max_length:
        log.info(f"TRUNCATING from {len(content)} to {max_length}")
        content = content[:max_length] + "\n\n...(truncated)"
    else:
        log.info("NOT truncating - content is under limit")

    return content


async def _fetch_from_service(url: str) -> str | None:
    """Call Docling Docker service to parse document.

    Supports two backends:
    - GPU service (docling-serve): /v1/convert API with CUDA acceleration
    - CPU service (custom): /parse API for fallback
    """
    # Try GPU service first if enabled
    if USE_DOCLING_GPU:
        result = await _fetch_from_gpu_service(url)
        if result is not None:
            return result
        log.debug("GPU service unavailable, falling back to CPU service")

    # Fall back to CPU service
    return await _fetch_from_cpu_service(url)


async def _fetch_from_gpu_service(url: str) -> str | None:
    """Call GPU Docling service (docling-serve) with /v1/convert API."""
    try:
        async with httpx.AsyncClient(timeout=DOCUMENT_TIMEOUT) as client:
            # docling-serve uses /v1/convert/source endpoint with sources array
            # Options optimized for LLM consumption:
            # - image_export_mode: "placeholder" avoids huge base64 blobs
            # - do_picture_description: true uses VLM to describe images
            # - picture_description_api: remote VLM endpoint (bluefin via Tailscale)
            response = await client.post(
                f"{DOCLING_GPU_URL}/v1/convert/source",
                json={
                    "sources": [{"kind": "http", "url": url}],
                    "options": {
                        "to_formats": ["md"],
                        "image_export_mode": "placeholder",
                        "do_picture_description": True,
                        "picture_description_api": {
                            "url": "http://master.tail5bb17d.ts.net:1234/v1/chat/completions",
                            "params": {
                                "model": "qwen/qwen3-vl-4b",
                                "max_completion_tokens": 500,
                            },
                            "timeout": 60,
                        },
                    }
                }
            )
            if response.status_code == 200:
                data = response.json()
                # docling-serve returns {"document": {"md_content": "..."}} or similar
                content = _extract_gpu_content(data)
                if content:
                    log.info(f"Parsed document via GPU service: {len(content)} chars")
                    return content
                else:
                    log.debug("GPU service returned empty content")
                    return None
            else:
                log.debug(f"GPU Docling service returned {response.status_code}")
                return None
    except httpx.ConnectError:
        log.debug("GPU Docling service not available")
        return None
    except Exception as e:
        log.debug(f"GPU Docling service error: {e}")
        return None


def _extract_gpu_content(data: dict) -> str | None:
    """Extract markdown content from docling-serve response.

    The response format can vary, so we check multiple possible locations.
    """
    # Try common response structures from docling-serve
    if "document" in data:
        doc = data["document"]
        # Check for md_content field
        if isinstance(doc, dict):
            if "md_content" in doc:
                return doc["md_content"]
            if "content" in doc:
                return doc["content"]
            if "markdown" in doc:
                return doc["markdown"]

    # Direct content field
    if "content" in data:
        return data["content"]
    if "md_content" in data:
        return data["md_content"]
    if "markdown" in data:
        return data["markdown"]

    # If data is a string directly
    if isinstance(data, str):
        return data

    return None


async def _fetch_from_cpu_service(url: str) -> str | None:
    """Call CPU Docling service (custom server.py) with /parse API."""
    try:
        async with httpx.AsyncClient(timeout=DOCUMENT_TIMEOUT) as client:
            response = await client.post(
                f"{DOCLING_URL}/parse",
                json={"url": url}
            )
            if response.status_code == 200:
                data = response.json()
                log.info(f"Parsed document via CPU service: {data.get('pages', 0)} pages")
                return data.get("content", "")
            else:
                log.debug(f"CPU Docling service returned {response.status_code}")
                return None
    except httpx.ConnectError:
        log.debug("CPU Docling service not available, trying local")
        return None
    except Exception as e:
        log.debug(f"CPU Docling service error: {e}")
        return None


async def _fetch_local(url: str) -> str | None:
    """Parse document using local Docling installation."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        log.debug("Local Docling not installed")
        return None

    try:
        file_path = await _download(url)
        log.info(f"Parsing document locally: {file_path.name}")

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        content = result.document.export_to_markdown()

        log.debug(f"Parsed document: {len(content)} characters")
        return content

    except Exception as e:
        log.error(f"Local document parsing failed: {e}")
        return None


async def _download(url: str) -> Path:
    """Download file to temp cache."""
    url_path = url.split("?")[0]
    suffix = Path(url_path).suffix or ".pdf"
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "docling_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{url_hash}{suffix}"

    if cache_path.exists():
        return cache_path

    async with httpx.AsyncClient(timeout=DOCUMENT_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        cache_path.write_bytes(response.content)

    return cache_path
