"""SQLite cache for parsed documents and chunks.

Provides persistent storage for:
- Full DoclingDocument JSON (avoids re-parsing expensive PDFs)
- Pre-computed chunks with headings and token counts
- Metadata for TOC generation
"""

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .config import CACHE_DB_PATH
from .logger import get_logger

log = get_logger("cache")


@dataclass
class CachedChunk:
    """A single chunk from a cached document."""
    chunk_id: int
    heading: str
    text: str
    token_count: int


@dataclass
class CachedDocument:
    """A cached document with its chunks."""
    url_hash: str
    url: str
    title: str
    full_markdown: str  # VLM-cleaned article markdown (for serving)
    total_tokens: int
    chunks: list[CachedChunk]
    raw_content: str = ""  # Original browser text dump (for debugging/re-processing)

    def get_full_text(self) -> str:
        """Get the full document text.

        Returns the VLM-cleaned markdown if available, otherwise
        reassembles from chunks.
        """
        if self.full_markdown:
            return self.full_markdown
        # Fallback: reassemble from chunks
        return "\n\n".join(chunk.text for chunk in self.chunks)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    url_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    doc_json TEXT,
    raw_content TEXT,
    total_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    url_hash TEXT,
    chunk_id INTEGER,
    heading TEXT,
    text TEXT,
    token_count INTEGER,
    PRIMARY KEY (url_hash, chunk_id),
    FOREIGN KEY (url_hash) REFERENCES documents(url_hash) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_url_hash ON chunks(url_hash);
"""

# Track if schema has been initialized this session
_schema_initialized = False


def _get_db_path() -> Path:
    """Get the database path, creating parent directories if needed."""
    db_path = Path(CACHE_DB_PATH).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def _db_connection() -> Iterator[sqlite3.Connection]:
    """Context manager for database connections with auto-initialization.

    Handles connection setup, schema initialization (once per session),
    and cleanup automatically.
    """
    global _schema_initialized

    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    if not _schema_initialized:
        conn.executescript(_SCHEMA)
        # Migration: add raw_content column if missing (for existing databases)
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN raw_content TEXT")
            log.info("Migrated database: added raw_content column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
        _schema_initialized = True

    try:
        yield conn
    finally:
        conn.close()


def url_to_hash(url: str) -> str:
    """Generate a stable hash for a URL."""
    return hashlib.md5(url.encode()).hexdigest()


def get_document(url: str) -> Optional[CachedDocument]:
    """Retrieve a cached document with all its chunks.

    Args:
        url: The document URL

    Returns:
        CachedDocument if found, None otherwise
    """
    url_hash = url_to_hash(url)

    try:
        with _db_connection() as conn:
            doc_row = conn.execute(
                "SELECT * FROM documents WHERE url_hash = ?",
                (url_hash,)
            ).fetchone()

            if not doc_row:
                return None

            chunk_rows = conn.execute(
                "SELECT * FROM chunks WHERE url_hash = ? ORDER BY chunk_id",
                (url_hash,)
            ).fetchall()

        chunks = [
            CachedChunk(
                chunk_id=row["chunk_id"],
                heading=row["heading"],
                text=row["text"],
                token_count=row["token_count"]
            )
            for row in chunk_rows
        ]

        log.debug(f"Cache hit for {url}: {len(chunks)} chunks")

        # Handle raw_content column (may not exist in older databases)
        try:
            raw_content = doc_row["raw_content"] or ""
        except (IndexError, KeyError):
            raw_content = ""

        return CachedDocument(
            url_hash=doc_row["url_hash"],
            url=doc_row["url"],
            title=doc_row["title"] or "",
            full_markdown=doc_row["doc_json"] or "",
            total_tokens=doc_row["total_tokens"] or 0,
            chunks=chunks,
            raw_content=raw_content
        )

    except sqlite3.Error as e:
        log.error(f"Database error getting document: {e}")
        return None


def save_document(
    url: str,
    title: str,
    full_markdown: str,
    total_tokens: int,
    chunks: list[dict],
    raw_content: str = ""
) -> bool:
    """Save a document and its chunks to cache.

    Args:
        url: The document URL
        title: Document title
        full_markdown: VLM-cleaned article markdown (for serving)
        total_tokens: Total token count for the document
        chunks: List of chunk dicts with keys: heading, text, token_count
        raw_content: Original browser text dump (for debugging/re-processing)

    Returns:
        True if saved successfully, False otherwise
    """
    url_hash = url_to_hash(url)

    try:
        with _db_connection() as conn:
            # Delete existing data (explicit delete for reliability)
            conn.execute("DELETE FROM chunks WHERE url_hash = ?", (url_hash,))
            conn.execute("DELETE FROM documents WHERE url_hash = ?", (url_hash,))

            conn.execute(
                """INSERT INTO documents (url_hash, url, title, doc_json, raw_content, total_tokens)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (url_hash, url, title, full_markdown, raw_content, total_tokens)
            )

            for i, chunk in enumerate(chunks):
                conn.execute(
                    """INSERT INTO chunks (url_hash, chunk_id, heading, text, token_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (url_hash, i, chunk["heading"], chunk["text"], chunk["token_count"])
                )

            conn.commit()

        log.info(f"Cached document {url}: {len(chunks)} chunks, {total_tokens} tokens")
        return True

    except sqlite3.Error as e:
        log.error(f"Database error saving document: {e}")
        return False


def get_chunk(url: str, chunk_id: int) -> Optional[CachedChunk]:
    """Retrieve a single chunk by ID.

    Args:
        url: The document URL
        chunk_id: The chunk index

    Returns:
        CachedChunk if found, None otherwise
    """
    url_hash = url_to_hash(url)

    try:
        with _db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE url_hash = ? AND chunk_id = ?",
                (url_hash, chunk_id)
            ).fetchone()

        if not row:
            return None

        return CachedChunk(
            chunk_id=row["chunk_id"],
            heading=row["heading"],
            text=row["text"],
            token_count=row["token_count"]
        )

    except sqlite3.Error as e:
        log.error(f"Database error getting chunk: {e}")
        return None


def clear_cache() -> bool:
    """Clear all cached documents and chunks.

    Returns:
        True if cleared successfully, False otherwise
    """
    try:
        with _db_connection() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.commit()
        log.info("Cache cleared")
        return True
    except sqlite3.Error as e:
        log.error(f"Database error clearing cache: {e}")
        return False


def get_cache_stats() -> dict:
    """Get cache statistics.

    Returns:
        Dict with document_count, chunk_count, total_tokens
    """
    try:
        with _db_connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_tokens = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM documents"
            ).fetchone()[0]

        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "total_tokens": total_tokens
        }

    except sqlite3.Error as e:
        log.error(f"Database error getting stats: {e}")
        return {"document_count": 0, "chunk_count": 0, "total_tokens": 0}
