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
    MAX_TOKENS = 8191  # text-embedding-3-small / ada-002 limit

    def __init__(self):
        from openai import AsyncOpenAI
        import tiktoken

        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.embedding_model
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _truncate(self, text: str) -> str:
        """Truncate text to MAX_TOKENS so the API never rejects it."""
        tokens = self._enc.encode(text)
        if len(tokens) <= self.MAX_TOKENS:
            return text
        return self._enc.decode(tokens[: self.MAX_TOKENS])

    async def embed(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=self._truncate(text),
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
