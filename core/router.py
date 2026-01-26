"""URL analysis and routing logic.

Three-way routing:
- Images → Vision AI (description)
- Documents (PDF, Office) → Docling (GPU-accelerated parsing)
- Web pages → Browser + VLM extraction (handles JS-heavy sites)
"""

import re
from pathlib import Path
from urllib.parse import urlparse

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'})

# Document types that Docling handles well (native parsing, no JS needed)
DOCUMENT_EXTENSIONS = frozenset({
    '.pdf',                           # PDFs
    '.doc', '.docx',                  # Word
    '.ppt', '.pptx',                  # PowerPoint
    '.xls', '.xlsx',                  # Excel
    '.odt', '.ods', '.odp',           # OpenDocument
    '.rtf',                           # Rich Text
    '.epub',                          # E-books
    '.md', '.txt',                    # Plain text/markdown
})

# URL patterns that indicate PDF even without extension
# e.g., arxiv.org/pdf/1706.03762 or semanticscholar.org/paper/xyz/pdf
PDF_URL_PATTERNS = [
    r'arxiv\.org/pdf/',               # arXiv PDFs
    r'/pdf$',                         # URLs ending in /pdf
    r'/pdf/',                         # URLs with /pdf/ in path
    r'\.pdf\?',                       # .pdf with query params
]


def _get_extension(url: str) -> str:
    """Extract file extension from URL."""
    path = urlparse(url).path.split("?")[0]
    return Path(path).suffix.lower()


def _is_pdf_url(url: str) -> bool:
    """Check if URL matches known PDF patterns (without extension)."""
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in PDF_URL_PATTERNS)


def route(url: str) -> str:
    """
    Determine which handler to use for a URL.

    Returns: 'image', 'document', or 'webpage'

    Three-way routing:
    - Images (.png, .jpg, etc.) → Vision AI (description)
    - Documents (.pdf, .docx, etc.) → Docling (GPU parsing, TOC/chunks)
    - Web pages (everything else) → Browser + VLM extraction
      - Renders JavaScript
      - Screenshot + text → VLM → clean markdown
      - Then local chunking for TOC/chunk navigation
    """
    ext = _get_extension(url)

    # Images go to vision handler
    if ext in IMAGE_EXTENSIONS:
        return "image"

    # Documents go to Docling (native PDF/Office parsing)
    if ext in DOCUMENT_EXTENSIONS:
        return "document"

    # Check for PDF URL patterns (e.g., arxiv.org/pdf/1706.03762)
    if _is_pdf_url(url):
        return "document"

    # Everything else is a web page - use browser + VLM extraction
    # This handles: HTML, JS-heavy SPAs, dynamic content, etc.
    return "webpage"
