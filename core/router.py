"""URL analysis and routing logic."""

from urllib.parse import urlparse
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'})
DOCUMENT_EXTENSIONS = frozenset({'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.xls', '.ppt'})

# URL patterns that indicate PDFs (e.g., arxiv.org/pdf/...)
PDF_URL_PATTERNS = ('/pdf/', '/pdfs/', '.pdf')


def _get_extension(url: str) -> str:
    """Extract file extension from URL."""
    path = urlparse(url).path.split("?")[0]
    return Path(path).suffix.lower()


def _is_pdf_url(url: str) -> bool:
    """Check if URL looks like a PDF (even without .pdf extension)."""
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in PDF_URL_PATTERNS)


def route(url: str) -> str:
    """
    Determine which handler to use for a URL.

    Returns: 'image', 'document', or 'browser'
    """
    ext = _get_extension(url)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    # Check for PDF-like URLs without extension (e.g., arxiv.org/pdf/...)
    if _is_pdf_url(url):
        return "document"
    # All web pages use browser + vision for consistent quality
    return "browser"
