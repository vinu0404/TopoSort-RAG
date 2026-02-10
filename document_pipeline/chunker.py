"""
Structure-aware document chunker.

BUG-3 FIX: `_parse_sections()` and `chunk_document()` are both `async`
           since they await the LLM for structure analysis.  The original
           spec had `await` inside a sync method which is a SyntaxError.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

import tiktoken

from utils.llm_providers import BaseLLMProvider


class StructureAwareChunker:
    """Chunks documents while preserving structural boundaries."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        chunk_size: int = 1024,
    ):
        self.llm = llm_provider
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Top-level entry point.

        Parameters
        ----------
        document : {"content": str, "metadata": {"filename": str, "doc_type": str, "uploaded_at": str}}
        """
        chunks: List[Dict[str, Any]] = []

        sections = await self._parse_sections(document["content"])

        for section_idx, section in enumerate(sections):
            if section.get("type") == "table":
                chunks.append(self._create_table_chunk(section, document, section_idx))
            else:
                chunks.extend(self._chunk_section(section, document, section_idx))

        chunks = self._add_relationships(chunks)
        return chunks
    async def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """Use the LLM to identify document structure (no regex)."""
        prompt = f"""Analyse this document and identify its structure.

Document (first 5000 chars):
{content[:5000]}

Return JSON:
{{
    "sections": [
        {{"type": "section", "title": "Introduction", "content": "...", "level": 1}},
        {{"type": "table", "content": "...", "description": "Revenue by quarter"}},
        ...
    ]
}}
"""
        response = await self.llm.generate(
            prompt=prompt,
            output_schema={"sections": [{"type": "string", "title": "string", "content": "string"}]},
            temperature=0.1,
        )

        if isinstance(response, dict):
            return response.get("sections", [])
        return [{"type": "section", "title": "", "content": content}]


    def _chunk_section(
        self, section: Dict[str, Any], document: Dict[str, Any], section_idx: int
    ) -> List[Dict[str, Any]]:
        paragraphs = section.get("content", "").split("\n\n")
        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunks.append(
                    self._create_chunk(current_chunk, document, section, section_idx, len(chunks))
                )
                current_chunk = para
                current_tokens = para_tokens
            else:
                current_chunk += ("\n\n" + para) if current_chunk else para
                current_tokens += para_tokens

        if current_chunk:
            chunks.append(
                self._create_chunk(current_chunk, document, section, section_idx, len(chunks))
            )

        return chunks

    def _create_chunk(
        self,
        text: str,
        document: Dict[str, Any],
        section: Dict[str, Any],
        section_idx: int,
        chunk_idx: int,
    ) -> Dict[str, Any]:
        meta = document.get("metadata", {})
        chunk_id = str(uuid.uuid4())
        return {
            "text": text,
            "chunk_id": chunk_id,
            "metadata": {
                "filename": meta.get("filename", ""),
                "doc_type": meta.get("doc_type", ""),
                "section_title": section.get("title", ""),
                "section_level": section.get("level", 0),
                "section_idx": section_idx,
                "chunk_idx": chunk_idx,
                "is_table": False,
                "parent_chunk_id": None,
                "uploaded_at": meta.get("uploaded_at", ""),
            },
        }

    def _create_table_chunk(
        self, section: Dict[str, Any], document: Dict[str, Any], section_idx: int
    ) -> Dict[str, Any]:
        chunk = self._create_chunk(
            section.get("content", ""), document, section, section_idx, 0
        )
        chunk["metadata"]["is_table"] = True
        return chunk

    @staticmethod
    def _add_relationships(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """First chunk of each section is parent of subsequent ones."""
        section_parents: Dict[int, str] = {}
        for chunk in chunks:
            sidx = chunk["metadata"]["section_idx"]
            cidx = chunk["metadata"]["chunk_idx"]
            if cidx == 0:
                section_parents[sidx] = chunk["chunk_id"]
                chunk["metadata"]["parent_chunk_id"] = None
            else:
                chunk["metadata"]["parent_chunk_id"] = section_parents.get(sidx)
        return chunks
