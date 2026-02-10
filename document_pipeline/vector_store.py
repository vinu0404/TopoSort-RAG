"""
Vector store adapter — Qdrant.

"""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient 
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance, PointStruct

from config.settings import config

logger = logging.getLogger(__name__)

_vector_store: Optional["VectorStore"] = None


def get_vector_store() -> "VectorStore":
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


class VectorStore:
    """Async Qdrant wrapper."""

    def __init__(self, url: str | None = None):
        

        self.client = AsyncQdrantClient(url=url or config.qdrant_url)

    # ── collection management ───────────────────────────────────────────

    async def create_user_collection(self, user_id: str) -> str:
        

        collection_name = f"user_{user_id}_documents"

        collections = await self.client.get_collections()
        existing = [c.name for c in collections.collections]
        if collection_name in existing:
            return collection_name

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

        # Payload indexes
        for field, schema in [
            ("metadata.doc_type", "keyword"),
            ("metadata.date", "keyword"),
            ("metadata.section_title", "text"),
        ]:
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )

        logger.info("Created collection '%s'", collection_name)
        return collection_name

    # ── writes ──────────────────────────────────────────────────────────

    async def add_document(
        self,
        user_id: str,
        document: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        embed_fn,
    ) -> None:
        
        collection_name = f"user_{user_id}_documents"
        doc_uuid = str(uuid.uuid4())
        doc_embedding = await embed_fn(
            f"{document['filename']}: {document.get('description', '')}"
        )

        doc_point = PointStruct(
            id=doc_uuid,
            vector=doc_embedding,
            payload={
                "type": "document_entry",
                "doc_id": document.get("doc_id", doc_uuid),
                "filename": document["filename"],
                "description": document.get("description", ""),
                "doc_type": document.get("doc_type", ""),
                "uploaded_at": document.get("uploaded_at", ""),
                "total_chunks": len(chunks),
            },
        )

        chunk_points = []
        for chunk in chunks:
            embedding = await embed_fn(chunk["text"])
            cid = chunk.get("chunk_id", str(uuid.uuid4()))
            chunk_points.append(
                PointStruct(
                    id=cid,
                    vector=embedding,
                    payload={
                        "type": "chunk",
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                    },
                )
            )

        await self.client.upsert(
            collection_name=collection_name,
            points=[doc_point, *chunk_points],
        )

    async def search(
        self,
        collection: str,
        embedding: List[float],
        filters: Dict[str, Any] | None = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters) if filters else None

        response = await self.client.query_points(
            collection_name=collection,
            query=embedding,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "chunk_id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in response.points
        ]

    async def bm25_search(
        self,
        collection: str,
        query: str,
        filters: Dict[str, Any] | None = None,
        limit: int = 20,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Sparse (keyword) search using BM25 scoring.
        """
        
        chunk_condition = FieldCondition(
            key="type", match=MatchValue(value="chunk")
        )
        base_conditions = [chunk_condition]
        if filters:
            user_filter = self._build_filter(filters)
            if user_filter and user_filter.must:
                base_conditions.extend(user_filter.must)

        scroll_filter = Filter(must=base_conditions)
        all_chunks: List[Dict[str, Any]] = []
        next_offset = None
        batch_size = 100

        while True:
            records, next_offset = await self.client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                text = record.payload.get("text", "")
                if text:
                    all_chunks.append({
                        "chunk_id": str(record.id),
                        "text": text,
                        "metadata": record.payload.get("metadata", {}),
                    })
            if next_offset is None:
                break

        if not all_chunks:
            return []
        scored = self._bm25_rank(query, all_chunks, k1=k1, b=b)
        return scored[:limit]

    _TOKENIZE_RE = re.compile(r"[a-z0-9]+")

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return cls._TOKENIZE_RE.findall(text.lower())

    @classmethod
    def _bm25_rank(
        cls,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Score *docs* against *query* using Okapi BM25.

        Returns docs sorted descending by score with a ``score`` key added.
        """
        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return docs

        n = len(docs)
        doc_tokens: List[List[str]] = [cls._tokenize(d["text"]) for d in docs]
        doc_lengths = [len(t) for t in doc_tokens]
        avgdl = sum(doc_lengths) / n if n else 1.0
        df: Counter = Counter()
        for tokens in doc_tokens:
            unique = set(tokens)
            for qt in query_tokens:
                if qt in unique:
                    df[qt] += 1
        idf: Dict[str, float] = {}
        for qt in query_tokens:
            idf[qt] = math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)

        scored: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            tf_map: Counter = Counter(doc_tokens[idx])
            dl = doc_lengths[idx]
            score = 0.0
            for qt in query_tokens:
                tf = tf_map[qt]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                score += idf.get(qt, 0.0) * (numerator / denominator)

            scored.append({**doc, "score": round(score, 6)})

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored


    @staticmethod
    def _build_filter(filters: Dict[str, Any]):
        """Convert flat filter dict to Qdrant Filter model."""
        
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None
