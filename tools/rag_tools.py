"""
RAG tools
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools import tool
from document_pipeline.vector_store import get_vector_store
from document_pipeline.embedder import get_embedding_model
from agents.rag_agent.prompts import RAGPrompts
from utils.llm_providers import get_llm_provider
from config.settings import config
from tools.registry import ToolRegistry

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
        doc_id = result["chunk_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result["chunk_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    all_docs = {r["chunk_id"]: r for r in dense_results + sparse_results}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[doc_id] for doc_id, _ in ranked]



@tool("rag_agent")
async def vector_search(
    query: str,
    user_id: str,
    filters: Dict[str, Any],
    top_k: int = 10,
    *,
    _vector_store=None,
    _embedding_model=None,
) -> List[Dict[str, Any]]:
    """Dense vector search using embeddings."""


    store = _vector_store or get_vector_store()
    embed = _embedding_model or get_embedding_model()

    collection_name = f"user_{user_id}_documents"
    query_embedding = await embed.embed(query)

    results = await store.search(
        collection=collection_name,
        embedding=query_embedding,
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
    _vector_store=None,
) -> List[Dict[str, Any]]:
    """Sparse retrieval using BM25."""

    store = _vector_store or get_vector_store()
    collection_name = f"user_{user_id}_documents"
    return await store.bm25_search(
        collection=collection_name,
        query=query,
        filters=filters,
        limit=top_k,
        k1=1.5,
        b=0.75,
    )


@tool("rag_agent")
async def hybrid_search(
    query: str,
    user_id: str,
    filters: Dict[str, Any],
    top_k: int = 20,
    *,
    _registry=None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search — dense + sparse with RRF merging.
    """
 
    registry = _registry or ToolRegistry()
    _vector_search = registry.get_tool("vector_search", "rag_agent")
    _bm25_search = registry.get_tool("bm25_search", "rag_agent")

    dense_results = await _vector_search(query, user_id, filters, top_k)
    sparse_results = await _bm25_search(query, user_id, filters, top_k)

    merged = await reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    return merged[:top_k]


@tool("rag_agent")
async def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
    *,
    _llm=None,
) -> List[Dict[str, Any]]:
    """LLM-based reranking of retrieved chunks."""

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
    return [chunks[i] for i in valid]
