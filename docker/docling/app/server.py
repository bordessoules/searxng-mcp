"""
Minimal Docling HTTP service for document parsing.
Calls external VLM API for document understanding.
"""

import os
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Docling Service")

VLM_API_URL = os.getenv("VLM_API_URL", "http://host.docker.internal:1234/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen3-vl-8b")


class ParseRequest(BaseModel):
    url: str
    prompt: str | None = None


class ParseResponse(BaseModel):
    content: str
    pages: int = 0


@app.post("/parse", response_model=ParseResponse)
async def parse_document(req: ParseRequest):
    """Parse a document URL and return extracted text."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(req.url)

        content = result.document.export_to_markdown()
        pages = len(result.document.pages) if hasattr(result.document, 'pages') else 0

        return ParseResponse(content=content, pages=pages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
