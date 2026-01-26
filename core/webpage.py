"""
Web page handler using Browser + VLM extraction pipeline.

Pipeline:
1. Browser (Playwright) renders JS and captures screenshot + text
2. VLM converts screenshot + text into clean article markdown
3. Local chunking creates TOC/chunk structure
4. Cache stores both raw text and VLM article for debugging/re-processing

VLM model is configurable via SUMMARIZE_MODEL env var.
This handles JS-heavy sites (SPAs, dynamic content) that Docling can't parse.
"""

import json
import re
from typing import Optional

from . import cache
from .browser import browse
from .config import CHUNK_SIZE_TOKENS, CHARS_PER_TOKEN, DOCUMENT_MAX_LENGTH
from .logger import get_logger
from .summarizer import extract_article_with_vlm, is_enabled

log = get_logger("webpage")


async def fetch(
    url: str,
    prompt: str | None = None,
    max_length: int = DOCUMENT_MAX_LENGTH,
    mode: str = "toc",
    chunk_id: int | None = None
) -> str:
    """
    Fetch a web page using browser + VLM extraction pipeline.

    Uses hybrid caching: any mode triggers caching, subsequent requests
    for any mode are served from cache instantly.

    Supports three modes for handling large pages:
    - "full": Complete article markdown (up to max_length)
    - "toc": Table of contents with chunk metadata (~2-5KB)
    - "chunk": Specific chunk by ID with navigation (~16-20KB)

    Args:
        url: URL to fetch
        prompt: Ignored (web pages use VLM extraction, not custom prompts)
        max_length: Max characters to return (for full mode)
        mode: Retrieval mode - "full", "toc", or "chunk"
        chunk_id: Required when mode="chunk", the chunk index to retrieve

    Returns:
        Page content (format depends on mode)
    """
    # Validate mode
    if mode not in ("full", "toc", "chunk"):
        return f"Error: Invalid mode '{mode}'. Use 'full', 'toc', or 'chunk'."

    if mode == "chunk" and chunk_id is None:
        return "Error: chunk_id is required when mode='chunk'."

    # Check cache first (for ALL modes - hybrid approach)
    cached_doc = cache.get_document(url)

    if cached_doc is None:
        # Not cached - fetch via browser, extract with VLM, chunk, and cache
        log.info(f"Cache miss for {url}, fetching via browser + VLM...")
        cached_doc = await _fetch_and_cache_webpage(url)

        if cached_doc is None:
            return "Error: Failed to fetch web page. Browser unavailable or VLM extraction failed."

    # Return appropriate response based on mode
    if mode == "toc":
        return _build_toc_response(cached_doc)
    elif mode == "chunk":
        return _get_chunk_response(cached_doc, chunk_id)
    else:  # mode == "full"
        content = cached_doc.get_full_text()

        if len(content) > max_length:
            content = content[:max_length] + "\n\n...(truncated)"

        return content


async def _fetch_and_cache_webpage(url: str) -> Optional[cache.CachedDocument]:
    """
    Fetch web page via browser, extract with VLM, chunk, and cache.

    This is the core pipeline:
    1. Browser renders JS and captures screenshot + text dump
    2. VLM converts screenshot + text into clean article markdown
    3. Local chunking creates heading-aware chunks
    4. Cache stores both raw text and VLM article

    Returns:
        CachedDocument if successful, None otherwise
    """
    # 1. Fetch via browser (renders JS)
    log.info(f"Browsing {url}...")
    result = await browse(url)

    if "error" in result and result["error"]:
        log.error(f"Browser error: {result['error']}")
        return None

    text_dump = result.get("content", "")
    screenshot = result.get("screenshot", "")
    final_url = result.get("url", url)

    if not text_dump and not screenshot:
        log.error("Browser returned no content")
        return None

    log.info(f"Got {len(text_dump)} chars text, screenshot={'yes' if screenshot else 'no'}")

    # 2. VLM extraction: convert to clean article markdown
    if is_enabled() and (text_dump or screenshot):
        log.info("Extracting article with VLM...")
        article_markdown = await extract_article_with_vlm(
            text_content=text_dump,
            screenshot_b64=screenshot,
            url=final_url
        )

        if not article_markdown or article_markdown.startswith("Summarization error"):
            log.warning(f"VLM extraction failed: {article_markdown[:100] if article_markdown else 'empty'}")
            # Fallback to raw text dump
            article_markdown = text_dump
    else:
        # No VLM available - use raw text dump
        log.warning("VLM not enabled, using raw text dump")
        article_markdown = text_dump

    if not article_markdown:
        log.error("No content extracted")
        return None

    # 3. Extract title and chunk locally
    title = _extract_title(article_markdown)
    chunks = _chunk_content(article_markdown)
    log.info(f"Chunked article: {len(chunks)} chunks")

    # Calculate total tokens
    total_tokens = sum(c["token_count"] for c in chunks)

    # 4. Save to cache (store BOTH raw text and VLM article)
    cache.save_document(
        url=final_url,
        title=title,
        full_markdown=article_markdown,  # VLM-cleaned article
        total_tokens=total_tokens,
        chunks=chunks,
        raw_content=text_dump  # Original browser text dump
    )

    # Return the cached document
    return cache.get_document(final_url)


# =============================================================================
# Chunking helpers (same logic as document.py for consistency)
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using character count."""
    return int(len(text) / CHARS_PER_TOKEN)


def _make_chunk(heading: str, text: str) -> dict:
    """Create a chunk dictionary with consistent structure."""
    return {
        "heading": heading,
        "text": text.strip(),
        "token_count": _estimate_tokens(text)
    }


def _split_large_section(section: str, heading: str, max_chars: int) -> list[dict]:
    """Split a large section into multiple chunks by paragraphs."""
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
    """Extract page title from content."""
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

    return "Untitled Page"


# =============================================================================
# Response builders (same interface as document.py)
# =============================================================================

PREVIEW_LENGTH = 120


def _extract_section_preview(text: str, max_length: int = PREVIEW_LENGTH) -> str:
    """Extract a clean preview from chunk text."""
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
    """Simplify chunk_ids for single-chunk sections."""
    chunk_ids = section["chunk_ids"]
    if len(chunk_ids) == 1:
        section["chunk_id"] = chunk_ids[0]
        del section["chunk_ids"]
    else:
        section["first_chunk"] = chunk_ids[0]
        section["last_chunk"] = chunk_ids[-1]


def _build_toc_response(doc: cache.CachedDocument) -> str:
    """Build table of contents response with grouped sections."""
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
    """Get a specific chunk with navigation context."""
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
