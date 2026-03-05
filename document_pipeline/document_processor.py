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
    """Generate a search-optimised LLM description for the document.
    
    The description is embedded and used as a retrieval target in
    stage-1 (document-level) search, so it must be keyword-rich and
    anticipate the kinds of queries users will ask.
    """
    llm = get_llm_provider(config.rag_model_provider, default_model=config.rag_model)
    prompt = f"""Generate a search-optimised description for this document.
This description will be used to match user queries, so be specific and keyword-rich.

Include:
1. Document type, format, and purpose (e.g. report, contract, spreadsheet).
2. ALL key entities: person names, organisation names, dates, product names,
   codes, acronyms, identifiers.
3. Main topics using specific terminology (avoid generic phrases).
4. Scope: time ranges, geographic regions, departments, or projects mentioned.
5. Key metrics, figures, or conclusions visible in the content.

Use terms a user would naturally search for.  Keep it to 4-6 dense sentences.

Filename: {filename}

Content (first 3000 characters):
{content[:3000]}

Description:
"""
    result = await llm.generate(prompt=prompt, max_tokens=200, temperature=0.3)
    return result.text.strip()


async def process_document(
    user_id: str,
    file_path: str,
    file_bytes: bytes | None = None,
) -> Dict[str, Any]:
    """
    Full pipeline: parse → (describe ‖ chunk) → batch-embed → store.
    """

    parsed = await parse_document(file_path, file_bytes)
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