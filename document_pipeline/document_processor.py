"""
Document processor – orchestrates parse → chunk → embed → store,
including LLM-generated document descriptions.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict

from config.settings import config
from utils.llm_providers import get_llm_provider
from document_pipeline.parser import parse_document
from document_pipeline.chunker import StructureAwareChunker
from document_pipeline.embedder import get_embedding_model
from document_pipeline.vector_store import get_vector_store

logger = logging.getLogger(__name__)


async def generate_document_description(filename: str, content: str) -> str:
    """Generate a 2-3 sentence LLM description for the document."""
    llm = get_llm_provider(config.rag_model_provider, default_model=config.rag_model)
    prompt = f"""Analyse this document and provide a concise description (2-3 sentences) covering:
1. Document type and purpose
2. Key topics covered
3. Time period or scope

Filename: {filename}

Content (first 2000 characters):
{content[:2000]}

Description:
"""
    response = await llm.generate(prompt=prompt, max_tokens=100, temperature=0.3)
    return response.strip() if isinstance(response, str) else str(response).strip()


async def process_document(
    user_id: str,
    file_path: str,
    file_bytes: bytes | None = None,
) -> Dict[str, Any]:
    """
    Full pipeline: parse → (describe ‖ chunk) → batch-embed → store.
    """

    parsed = await parse_document(file_path, file_bytes)

    # ── Run description + chunking in parallel (independent) ────────
    chunker = StructureAwareChunker(chunk_size=1024)
    description_task = generate_document_description(
        parsed["metadata"]["filename"],
        parsed["content"],
    )
    chunk_task = chunker.chunk_document(parsed)
    description, chunks = await asyncio.gather(description_task, chunk_task)

    logger.info(
        "Parsed '%s': %d chunks, description generated",
        parsed["metadata"]["filename"],
        len(chunks),
    )

    store = get_vector_store()
    await store.create_user_collection(user_id)
    embed_model = get_embedding_model()

    doc_record = {
        "doc_id": str(uuid.uuid4()),
        "filename": parsed["metadata"]["filename"],
        "description": description,
        "doc_type": parsed["metadata"]["doc_type"],
        "uploaded_at": parsed["metadata"]["uploaded_at"],
    }

    # Pass both embed and embed_batch — store uses batch for speed
    await store.add_document(
        user_id=user_id,
        document=doc_record,
        chunks=chunks,
        embed_fn=embed_model.embed,
        embed_batch_fn=embed_model.embed_batch,
    )

    return {
        "doc_id": doc_record["doc_id"],
        "filename": doc_record["filename"],
        "total_chunks": len(chunks),
        "description": description,
    }