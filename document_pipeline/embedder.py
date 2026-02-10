"""Embedding model adapter."""

from __future__ import annotations

from typing import List, Optional

from config.settings import config

_embedding_model: Optional["EmbeddingModel"] = None


def get_embedding_model() -> "EmbeddingModel":
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


class EmbeddingModel:
    def __init__(self):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.embedding_model

    async def embed(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
