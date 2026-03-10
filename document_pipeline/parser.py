"""Document parsing — PDF, DOCX, Excel, Images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import pymupdf  
import docx
from io import BytesIO
import pandas as pd
logger = logging.getLogger(__name__)

# Image extensions handled by the vision LLM
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


async def parse_document(file_path: str, file_bytes: bytes | None = None) -> Dict[str, Any]:
    """
    Extract text content from a document.

    Returns {"content": str, "metadata": {...}}
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        content = _parse_pdf(file_path, file_bytes)
    elif ext in (".docx", ".doc"):
        content = _parse_docx(file_path, file_bytes)
    elif ext in (".xlsx", ".xls", ".csv"):
        content = _parse_spreadsheet(file_path, file_bytes)
    elif ext in _IMAGE_EXTENSIONS:
        content = await _parse_image(file_path, file_bytes)
    elif ext in (".txt", ".md"):
        content = _parse_text(file_path, file_bytes)
    else:
        content = _parse_text(file_path, file_bytes)

    return {
        "content": content,
        "metadata": {
            "filename": Path(file_path).name,
            "doc_type": ext.lstrip("."),
            "uploaded_at": "",
        },
    }


# Sentinel injected between PDF pages so the chunker can track page numbers.
_PAGE_MARKER = "<!-- PAGE {} -->"


def _parse_pdf(path: str, data: bytes | None) -> str:
    try:
        if data:
            doc = pymupdf.open(stream=data, filetype="pdf")
        else:
            doc = pymupdf.open(path)
        parts: list[str] = []
        for page_num, page in enumerate(doc, start=1):
            parts.append(_PAGE_MARKER.format(page_num))
            parts.append(page.get_text())
        return "\n\n".join(parts)
    except ImportError:
        logger.warning("pymupdf not installed — returning empty content for PDF")
        return ""


def _parse_docx(path: str, data: bytes | None) -> str:
    try:
        
        source = BytesIO(data) if data else path
        doc = docx.Document(source)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.warning("python-docx not installed — returning empty content for DOCX")
        return ""


def _parse_spreadsheet(path: str, data: bytes | None) -> str:
    try:
        source = BytesIO(data) if data else path
        df = pd.read_excel(source) if not path.endswith(".csv") else pd.read_csv(source)
        return df.to_string()
    except ImportError:
        logger.warning("pandas not installed — returning empty content for spreadsheet")
        return ""


def _parse_text(path: str, data: bytes | None) -> str:
    if data:
        return data.decode(errors="replace")
    return Path(path).read_text(errors="replace")


async def _parse_image(path: str, data: bytes | None) -> str:
    """Send image to the configured vision LLM and return text description."""
    from document_pipeline.vision import analyze_image

    if not data:
        data = Path(path).read_bytes()
    filename = Path(path).name
    logger.info("Parsing image via vision LLM: %s (%d bytes)", filename, len(data))
    return await analyze_image(data, filename)
