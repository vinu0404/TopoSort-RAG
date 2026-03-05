"""
RAG tools — two-level retrieval pipeline.

Stage 1: description-level dense search → candidate doc_ids
Stage 2: chunk-level hybrid (dense ‖ BM25) within matched docs → RRF
Optional: LLM reranking (controlled by config.rag_use_llm_reranking)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from tools import tool
from document_pipeline.vector_store import get_vector_store
from document_pipeline.embedder import get_embedding_model
from agents.rag_agent.prompts import RAGPrompts
from utils.llm_providers import get_llm_provider
from config.settings import config

logger = logging.getLogger(__name__)


# ── helpers (not tools) ─────────────────────────────────────────────────

async def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.  score(doc) = Σ 1/(k + rank)
    """
    scores: Dict[str, float] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    all_docs = {r["chunk_id"]: r for r in dense_results + sparse_results}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[chunk_id] for chunk_id, _ in ranked]


# ── atomic search tools ─────────────────────────────────────────────────

@tool("rag_agent")
async def vector_search(
    query: str,
    user_id: str,
    filters: Dict[str, Any],
    top_k: int = 10,
    *,
    doc_ids: Optional[List[str]] = None,
    _vector_store=None,
    _embedding_model=None,
) -> List[Dict[str, Any]]:
    """Dense vector search — **chunks only**, optionally scoped to doc_ids."""

    store = _vector_store or get_vector_store()
    embed = _embedding_model or get_embedding_model()

    collection_name = f"user_{user_id}_documents"
    query_embedding = await embed.embed(query)

    results = await store.search_chunks(
        collection=collection_name,
        embedding=query_embedding,
        doc_ids=doc_ids,
        filters=filters,
        limit=top_k,
    )
    return results


@tool("rag_agent")
async def bm25_search(
    query: str,
    user_id: str,
    filters: Dict[str, Any],
    top_k: int = 20,
    *,
    doc_ids: Optional[List[str]] = None,
    _vector_store=None,
) -> List[Dict[str, Any]]:
    """Sparse BM25 search — scoped to doc_ids when provided (two-level stage 2)."""

    store = _vector_store or get_vector_store()
    collection_name = f"user_{user_id}_documents"

    if doc_ids:
        return await store.bm25_search_by_doc_ids(
            collection=collection_name,
            query=query,
            doc_ids=doc_ids,
            filters=filters,
            limit=top_k,
        )
    # Fallback: full-collection BM25 (legacy path)
    return await store.bm25_search(
        collection=collection_name,
        query=query,
        filters=filters,
        limit=top_k,
    )


@tool("rag_agent")
async def hybrid_search(
    query: str,
    user_id: str,
    filters: Dict[str, Any],
    top_k: int = 20,
    *,
    doc_ids: Optional[List[str]] = None,
    _vector_store=None,
    _embedding_model=None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search — dense + BM25 with RRF merging.
    Both branches are scoped to *doc_ids* when supplied (two-level stage 2).
    Runs dense and sparse searches **in parallel**.
    """

    dense_coro = vector_search(
        query=query,
        user_id=user_id,
        filters=filters,
        top_k=top_k,
        doc_ids=doc_ids,
        _vector_store=_vector_store,
        _embedding_model=_embedding_model,
    )
    sparse_coro = bm25_search(
        query=query,
        user_id=user_id,
        filters=filters,
        top_k=top_k,
        doc_ids=doc_ids,
        _vector_store=_vector_store,
    )

    dense_results, sparse_results = await asyncio.gather(dense_coro, sparse_coro)

    merged = await reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    return merged[:top_k]


# ── two-level orchestration ─────────────────────────────────────────────

@tool("rag_agent")
async def two_level_search(
    query: str,
    user_id: str,
    filters: Dict[str, Any],
    top_k: int = 20,
    *,
    _vector_store=None,
    _embedding_model=None,
) -> Dict[str, Any]:
    """
    Two-level retrieval:

    Stage 1 — embed *query*, search ``type=document_entry`` → top-N doc_ids.
    Stage 2 — hybrid search (dense ‖ BM25 in parallel) scoped to those doc_ids.
    Fallback — if Stage 1 returns too few documents above the score threshold,
               also run an unscoped chunk search and merge results.

    Returns ``{"chunks": [...], "matched_documents": [...]}``.
    """

    store = _vector_store or get_vector_store()
    embed_model = _embedding_model or get_embedding_model()
    collection_name = f"user_{user_id}_documents"

    # ── Stage 1: description-level search ───────────────────────────────
    query_embedding = await embed_model.embed(query)

    description_top_k = config.rag_description_top_k
    score_threshold = config.rag_description_score_threshold

    doc_results = await store.search_documents(
        collection=collection_name,
        embedding=query_embedding,
        filters=filters,
        limit=description_top_k,
    )

    # Keep only docs above the dynamic threshold
    strong_docs = [d for d in doc_results if d["score"] >= score_threshold]
    doc_ids = [d["doc_id"] for d in strong_docs]

    logger.info(
        "[two_level_search] Stage-1: %d/%d docs above threshold %.2f — doc_ids=%s",
        len(strong_docs),
        len(doc_results),
        score_threshold,
        doc_ids,
    )

    # ── Stage 2: scoped hybrid search ───────────────────────────────────
    needs_fallback = len(doc_ids) < 2  # too few confident matches

    if doc_ids:
        scoped_coro = hybrid_search(
            query=query,
            user_id=user_id,
            filters=filters,
            top_k=top_k,
            doc_ids=doc_ids,
            _vector_store=store,
            _embedding_model=embed_model,
        )
    else:
        scoped_coro = None

    if needs_fallback:
        fallback_coro = hybrid_search(
            query=query,
            user_id=user_id,
            filters=filters,
            top_k=top_k,
            doc_ids=None,  # unscoped
            _vector_store=store,
            _embedding_model=embed_model,
        )
    else:
        fallback_coro = None

    # Run whichever coroutines we need in parallel
    coros = [c for c in (scoped_coro, fallback_coro) if c is not None]
    results = await asyncio.gather(*coros)

    # Merge (deduplicate by chunk_id, scoped results take priority)
    seen_ids: set = set()
    merged: List[Dict[str, Any]] = []

    for result_list in results:
        for chunk in result_list:
            cid = chunk["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

    merged = merged[:top_k]

    logger.info(
        "[two_level_search] Stage-2: %d chunks (fallback=%s)",
        len(merged),
        needs_fallback,
    )

    return {
        "chunks": merged,
        "matched_documents": strong_docs,
    }


# ── LLM reranking ───────────────────────────────────────────────────────

@tool("rag_agent")
async def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
    *,
    _llm=None,
) -> List[Dict[str, Any]]:
    """LLM-based reranking of retrieved chunks.

    Skipped automatically when ``config.rag_use_llm_reranking`` is False
    — returns the first *top_k* chunks in their existing order.
    """

    if not config.rag_use_llm_reranking:
        return chunks[:top_k]

    llm = _llm or get_llm_provider(config.rag_model_provider)

    prompt = RAGPrompts.rerank_prompt(query, chunks)
    response = await llm.generate(
        prompt=prompt,
        output_schema={"ranked_indices": "List[int]"},
        temperature=0.1,
        model=config.rag_model,
    )

    ranked_indices = response.get("ranked_indices", [])[:top_k]
    valid = [i for i in ranked_indices if 0 <= i < len(chunks)]
    return [chunks[i] for i in valid] if valid else chunks[:top_k]
