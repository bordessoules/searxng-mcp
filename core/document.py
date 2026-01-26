"""Document parsing handler - calls Docling service or falls back to local.

Supports three retrieval modes for handling large documents:
- "full": Returns complete document (existing behavior, up to 1MB)
- "toc": Returns table of contents with chunk metadata
- "chunk": Returns a specific chunk by ID with navigation
"""

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from .config import (
    DOCLING_URL, DOCLING_GPU_URL, USE_DOCLING_GPU,
    DOCUMENT_TIMEOUT, DOCUMENT_MAX_LENGTH,
    CHUNK_SIZE_TOKENS, CHARS_PER_TOKEN,
    SUMMARIZE_API_URL, SUMMARIZE_MODEL, SUMMARIZE_TIMEOUT
)
from .logger import get_logger
from . import cache

log = get_logger("document")


async def check_gpu_service() -> dict:
    """Check if GPU Docling service is available.

    Returns a dict with status info for logging at startup.
    """
    result = {
        "gpu_available": False,
        "gpu_url": DOCLING_GPU_URL,
        "gpu_enabled": USE_DOCLING_GPU,
        "message": ""
    }

    if not USE_DOCLING_GPU:
        result["message"] = "GPU service disabled (USE_DOCLING_GPU=false)"
        return result

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{DOCLING_GPU_URL}/health")
            if response.status_code == 200:
                result["gpu_available"] = True
                result["message"] = f"GPU service ready at {DOCLING_GPU_URL} (with VLM: {SUMMARIZE_MODEL})"
            else:
                result["message"] = f"GPU service returned {response.status_code}"
    except httpx.ConnectError:
        result["message"] = f"GPU service not reachable at {DOCLING_GPU_URL}"
    except Exception as e:
        result["message"] = f"GPU service error: {e}"

    return result


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using character count.

    Uses a simple heuristic: ~4 characters per token on average.
    This avoids HuggingFace tokenizer dependency while being
    reasonably accurate for English text.
    """
    return int(len(text) / CHARS_PER_TOKEN)


async def fetch(
    url: str,
    prompt: str | None = None,
    max_length: int = DOCUMENT_MAX_LENGTH,
    mode: str = "toc",
    chunk_id: int | None = None
) -> str:
    """
    Parse a document from URL using Docling service.

    Uses a hybrid caching approach: any mode triggers caching, and subsequent
    requests for any mode are served from cache instantly.

    Supports three modes for handling large documents:
    - "full": Complete document (default, up to max_length)
    - "toc": Table of contents with chunk metadata (~2-5KB)
    - "chunk": Specific chunk by ID with navigation (~16-20KB)

    Args:
        url: URL to document (PDF, DOCX, etc.)
        prompt: Ignored (documents don't need summarization prompt)
        max_length: Max characters to return (for full mode)
        mode: Retrieval mode - "full", "toc", or "chunk"
        chunk_id: Required when mode="chunk", the chunk index to retrieve

    Returns:
        Parsed document content (format depends on mode)
    """
    # Validate mode
    if mode not in ("full", "toc", "chunk"):
        return f"Error: Invalid mode '{mode}'. Use 'full', 'toc', or 'chunk'."

    if mode == "chunk" and chunk_id is None:
        return "Error: chunk_id is required when mode='chunk'."

    # Check cache first (for ALL modes - hybrid approach)
    cached_doc = cache.get_document(url)

    if cached_doc is None:
        # Not cached - fetch, chunk, and cache the document
        log.info(f"Cache miss for {url}, fetching and caching...")
        cached_doc = await _fetch_and_cache_document(url)

        if cached_doc is None:
            return "Error: Failed to fetch document. Docling service unavailable and local Docling not installed."

    # Return appropriate response based on mode
    if mode == "toc":
        return _build_toc_response(cached_doc)
    elif mode == "chunk":
        return _get_chunk_response(cached_doc, chunk_id)
    else:  # mode == "full"
        content = cached_doc.get_full_text()

        log.info(f"Content length: {len(content)}, max_length: {max_length}")
        if len(content) > max_length:
            log.info(f"TRUNCATING from {len(content)} to {max_length}")
            content = content[:max_length] + "\n\n...(truncated)"
        else:
            log.info("NOT truncating - content is under limit")

        return content


async def _fetch_and_cache_document(url: str) -> Optional[cache.CachedDocument]:
    """Fetch document, chunk it, cache everything, and return cached doc.

    This is the core of the hybrid caching approach:
    1. Fetch the full document content (with VLM image descriptions)
    2. Chunk it locally for TOC/chunk navigation
    3. Store BOTH full markdown AND chunks in cache
    4. Return the cached document

    Priority: GPU convert (with VLM) → GPU chunking → CPU service → local.
    VLM model is configurable via SUMMARIZE_MODEL env var.
    Since we cache results, VLM cost is only paid once per document.
    """
    full_content = None

    # Try GPU convert endpoint FIRST - this uses VLM for image descriptions
    # Since we have caching, the VLM cost is only paid once per document
    log.info(f"Fetching document via GPU convert service (VLM: {SUMMARIZE_MODEL})...")
    full_content = await _fetch_from_gpu_service(url)

    if full_content is not None:
        log.info(f"Got {len(full_content)} chars with VLM image descriptions")
    else:
        # Fall back to GPU chunking endpoint (no VLM, but still fast)
        log.warning("GPU convert service unavailable, trying chunking endpoint...")
        chunks = await _fetch_chunks_from_service(url)

        if chunks is not None:
            full_content = "\n\n".join(c["text"] for c in chunks)
            log.info(f"Got {len(chunks)} chunks from GPU chunking service (no VLM)")
        else:
            # Fall back to CPU service
            log.warning("GPU services unavailable, trying CPU service...")
            full_content = await _fetch_from_cpu_service(url)

        if full_content is None:
            # Last resort: local CPU parsing (SLOW!)
            log.warning("All Docling services unavailable! Falling back to LOCAL CPU parsing.")
            log.warning("This will be SLOW (30-60+ seconds). Consider starting the GPU service:")
            log.warning("  cd docker/docling && docker-compose -f docker-compose.gpu.yml up -d")
            full_content = await _fetch_local(url)

    if full_content is None:
        return None

    # Extract title and chunk locally (structural chunking)
    title = _extract_title(full_content)
    chunks = _chunk_content(full_content)
    log.info(f"Chunked document locally: {len(chunks)} chunks")

    # Calculate total tokens
    total_tokens = sum(c["token_count"] for c in chunks)

    # Save to cache with full markdown
    cache.save_document(
        url=url,
        title=title,
        full_markdown=full_content,  # Store original for perfect reconstruction
        total_tokens=total_tokens,
        chunks=chunks
    )

    # Return the cached document
    return cache.get_document(url)


async def _fetch_chunks_from_service(url: str) -> Optional[list[dict]]:
    """Call Docling GPU service chunking endpoint.

    Uses /v1/chunk/hybrid/source for semantic chunking.
    """
    if not USE_DOCLING_GPU:
        return None

    try:
        async with httpx.AsyncClient(timeout=DOCUMENT_TIMEOUT) as client:
            response = await client.post(
                f"{DOCLING_GPU_URL}/v1/chunk/hybrid/source",
                json={
                    "sources": [{"kind": "http", "url": url}],
                    "options": {
                        "max_tokens": CHUNK_SIZE_TOKENS,
                        "include_metadata": True
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                chunks = _parse_service_chunks(data)
                if chunks:
                    log.info(f"Got {len(chunks)} chunks from GPU service")
                    return chunks
                else:
                    log.debug("GPU chunking returned empty result")
                    return None
            else:
                log.debug(f"GPU chunking service returned {response.status_code}")
                return None

    except httpx.ConnectError:
        log.warning("GPU chunking service not reachable at %s", DOCLING_GPU_URL)
        return None
    except httpx.TimeoutException:
        log.warning("GPU chunking service timed out (limit: %s seconds)", DOCUMENT_TIMEOUT)
        return None
    except Exception as e:
        log.warning(f"GPU chunking service error: {type(e).__name__}: {e}")
        return None


def _extract_chunks_data(data: dict) -> list | None:
    """Extract chunks array from various Docling response formats."""
    if "chunks" in data:
        return data["chunks"]
    if "document" in data and "chunks" in data["document"]:
        return data["document"]["chunks"]
    if isinstance(data, list):
        return data
    return None


def _extract_chunk_heading(chunk: dict, text: str) -> str:
    """Extract heading from a chunk, with fallbacks."""
    # API returns "headings" (list) - use the most specific (last) one
    headings = chunk.get("headings", [])
    if headings and isinstance(headings, list):
        return headings[-1]

    # Fallback to singular heading field
    heading = chunk.get("heading", chunk.get("title", ""))
    if heading:
        return heading

    # Last resort: extract from text
    return _extract_heading_from_text(text)


def _parse_service_chunks(data: dict) -> Optional[list[dict]]:
    """Parse chunks from Docling service response.

    The docling-serve API returns chunks with these fields:
    - text: The chunk content
    - headings: List of heading strings (e.g., ["1 Introduction", "1.1 Background"])
    - num_tokens: Token count
    - page_numbers: List of page numbers
    """
    chunks_data = _extract_chunks_data(data)
    if not chunks_data:
        return None

    chunks = []
    for chunk in chunks_data:
        text = chunk.get("text", chunk.get("content", ""))
        heading = _extract_chunk_heading(chunk, text)
        token_count = chunk.get("num_tokens", _estimate_tokens(text))

        chunks.append({
            "heading": heading,
            "text": text,
            "token_count": token_count
        })

    return chunks if chunks else None


def _make_chunk(heading: str, text: str) -> dict:
    """Create a chunk dictionary with consistent structure."""
    return {
        "heading": heading,
        "text": text.strip(),
        "token_count": _estimate_tokens(text)
    }


def _split_large_section(section: str, heading: str, max_chars: int) -> list[dict]:
    """Split a large section into multiple chunks by paragraphs.

    Keeps the same heading for all chunks (they're parts of the same section).
    """
    chunks = []
    paragraphs = section.split("\n\n")
    current_chunk = ""
    part_num = 1

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunk_heading = f"{heading} (part {part_num})" if part_num > 1 else heading
            chunks.append(_make_chunk(chunk_heading, current_chunk))
            current_chunk = ""
            part_num += 1

        current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunk_heading = f"{heading} (part {part_num})" if part_num > 1 else heading
        chunks.append(_make_chunk(chunk_heading, current_chunk))

    return chunks


def _chunk_content(content: str) -> list[dict]:
    """Chunk content locally using heading-aware splitting.

    Creates ONE CHUNK PER SECTION for fine-grained TOC navigation.
    Only splits further if a section exceeds the token limit.
    """
    max_chars = int(CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN)

    # Split by markdown headings (##, ###, etc.) - keep heading with content
    sections = re.split(r'(?m)(?=^#{1,6}\s)', content)

    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading from section
        heading_match = re.match(r'^(#{1,6}\s+.+?)(?:\n|$)', section)
        section_heading = heading_match.group(1).strip() if heading_match else ""

        # If section is too large, split it further
        if len(section) > max_chars:
            sub_chunks = _split_large_section(section, section_heading, max_chars)
            chunks.extend(sub_chunks)
        else:
            # One chunk per section (fine-grained TOC)
            heading = section_heading or f"Section {len(chunks) + 1}"
            chunks.append(_make_chunk(heading, section))

    # Handle case where content has no headings - split by size
    if len(chunks) <= 1 and len(content) > max_chars:
        chunks = _chunk_by_size(content, max_chars)

    return chunks


def _chunk_by_size(content: str, max_chars: int) -> list[dict]:
    """Split content into fixed-size chunks when no headings exist."""
    chunks = []
    paragraphs = content.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunks.append(_make_chunk(f"Part {len(chunks) + 1}", current_chunk))
            current_chunk = ""

        current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(_make_chunk(f"Part {len(chunks) + 1}", current_chunk))

    return chunks


def _extract_title(content: str) -> str:
    """Extract document title from content."""
    # Try to find first heading
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Try first non-empty line
    lines = content.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 200:
            return line

    return "Untitled Document"


def _extract_heading_from_text(text: str) -> str:
    """Extract a heading from chunk text."""
    # Look for markdown heading
    match = re.search(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Use first line if short enough
    first_line = text.split("\n")[0].strip()
    if first_line and len(first_line) < 100:
        return first_line[:80]

    return ""


PREVIEW_LENGTH = 120


def _extract_section_preview(text: str, max_length: int = PREVIEW_LENGTH) -> str:
    """Extract a clean preview from chunk text.

    Removes markdown headings and truncates at word boundary.
    """
    preview = text.strip()[:max_length]

    # Remove heading line if present
    if preview.startswith("#"):
        lines = text.strip().split("\n", 1)
        preview = lines[1].strip()[:max_length] if len(lines) > 1 else ""

    # Truncate at word boundary if at max length
    if len(preview) == max_length:
        preview = preview.rsplit(" ", 1)[0] + "..."

    return preview


def _finalize_section_chunk_ids(section: dict) -> None:
    """Simplify chunk_ids for single-chunk sections, add first/last for multi-chunk."""
    chunk_ids = section["chunk_ids"]
    if len(chunk_ids) == 1:
        section["chunk_id"] = chunk_ids[0]
        del section["chunk_ids"]
    else:
        section["first_chunk"] = chunk_ids[0]
        section["last_chunk"] = chunk_ids[-1]


def _build_toc_response(doc: cache.CachedDocument) -> str:
    """Build table of contents response with grouped sections.

    Groups consecutive chunks with the same heading into sections,
    providing a cleaner document outline for LLM navigation.

    Returns JSON with:
    - sections: List of {heading, tokens, chunk_ids, first_chunk, last_chunk}
    - total_tokens, total_chunks for overview
    """
    # Get preview from first chunk
    preview = ""
    if doc.chunks:
        first_text = doc.chunks[0].text
        preview = first_text[:200] + ("..." if len(first_text) > 200 else "")

    # Group consecutive chunks by heading
    sections = []
    current_section = None

    for chunk in doc.chunks:
        if current_section is None or chunk.heading != current_section["heading"]:
            if current_section is not None:
                sections.append(current_section)

            current_section = {
                "heading": chunk.heading,
                "preview": _extract_section_preview(chunk.text),
                "tokens": chunk.token_count,
                "chunk_ids": [chunk.chunk_id],
            }
        else:
            current_section["tokens"] += chunk.token_count
            current_section["chunk_ids"].append(chunk.chunk_id)

    if current_section is not None:
        sections.append(current_section)

    for section in sections:
        _finalize_section_chunk_ids(section)

    toc = {
        "mode": "toc",
        "source_url": doc.url,  # Citation tracking
        "title": doc.title,
        "preview": preview,
        "total_tokens": doc.total_tokens,
        "total_chunks": len(doc.chunks),
        "total_sections": len(sections),
        "sections": sections,
        "hint": "Use mode='chunk', chunk_id=N to retrieve a section. For multi-chunk sections, request each chunk_id in the range."
    }

    return json.dumps(toc, indent=2)


def _get_chunk_response(doc: cache.CachedDocument, chunk_id: int) -> str:
    """Get a specific chunk with navigation context.

    Returns JSON with chunk content and prev/next links.
    """
    if chunk_id < 0 or chunk_id >= len(doc.chunks):
        return json.dumps({
            "error": f"Invalid chunk_id {chunk_id}. Valid range: 0-{len(doc.chunks) - 1}"
        })

    chunk = doc.chunks[chunk_id]

    # Build breadcrumbs (simplified - just current heading for now)
    breadcrumbs = [chunk.heading]

    # Navigation
    prev_chunk = None
    next_chunk = None

    if chunk_id > 0:
        prev = doc.chunks[chunk_id - 1]
        prev_chunk = {"id": prev.chunk_id, "heading": prev.heading}

    if chunk_id < len(doc.chunks) - 1:
        next_c = doc.chunks[chunk_id + 1]
        next_chunk = {"id": next_c.chunk_id, "heading": next_c.heading}

    response = {
        "mode": "chunk",
        "source_url": doc.url,  # Citation tracking
        "title": doc.title,
        "chunk_id": chunk_id,
        "total_chunks": len(doc.chunks),
        "heading": chunk.heading,
        "breadcrumbs": breadcrumbs,
        "tokens": chunk.token_count,
        "text": chunk.text,
        "prev_chunk": prev_chunk,
        "next_chunk": next_chunk
    }

    return json.dumps(response, indent=2)


# =============================================================================
# Existing service fetching functions (unchanged)
# =============================================================================

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
            # Build VLM API URL for picture descriptions
            # Uses your LM Studio / OpenAI-compatible endpoint from config
            vlm_chat_url = SUMMARIZE_API_URL.rstrip('/') + '/chat/completions'

            response = await client.post(
                f"{DOCLING_GPU_URL}/v1/convert/source",
                json={
                    "sources": [{"kind": "http", "url": url}],
                    "options": {
                        "to_formats": ["md"],
                        "image_export_mode": "placeholder",
                        "do_picture_description": True,
                        "picture_description_api": {
                            "url": vlm_chat_url,
                            "params": {
                                "model": SUMMARIZE_MODEL,
                                "max_completion_tokens": 500,
                            },
                            "timeout": SUMMARIZE_TIMEOUT,
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
        log.warning("GPU Docling convert service not reachable at %s", DOCLING_GPU_URL)
        return None
    except httpx.TimeoutException:
        log.warning("GPU Docling convert service timed out (limit: %s seconds)", DOCUMENT_TIMEOUT)
        return None
    except Exception as e:
        log.warning(f"GPU Docling convert service error: {type(e).__name__}: {e}")
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
