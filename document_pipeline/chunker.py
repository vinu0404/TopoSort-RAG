"""
Structure-aware document chunker.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

import tiktoken
from utils.chunk_utils import SectionParser

class StructureAwareChunker:
    """Chunks documents while preserving structural boundaries."""

    def __init__(
        self,
        chunk_size: int = 1024,
    ):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.section_parser = SectionParser()

    async def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Top-level entry point.

        Parameters
        ----------
        document : {"content": str, "metadata": {"filename": str, "doc_type": str, "uploaded_at": str}}
        
        Returns
        -------
        List of chunk dicts with format:
        {
            "text": str,
            "chunk_id": str,
            "metadata": {
                "filename": str,
                "doc_type": str,
                "section_title": str,
                "section_level": int,
                "section_idx": int,
                "chunk_idx": int,
                "is_table": bool,
                "parent_chunk_id": str | None,
                "uploaded_at": str,
            }
        }
        """
        chunks: List[Dict[str, Any]] = []
        sections = await self._parse_sections(
            content=document["content"],
            doc_type=document.get("metadata", {}).get("doc_type", "pdf")
        )

        for section_idx, section in enumerate(sections):
            if section.get("type") == "table":
                chunks.append(self._create_table_chunk(section, document, section_idx))
            else:
                chunks.extend(self._chunk_section(section, document, section_idx))

        chunks = self._add_relationships(chunks)
        return chunks

    async def _parse_sections(
        self,
        content: str,
        doc_type: str = "pdf"
    ) -> List[Dict[str, Any]]:
        """
        Parse document sections using regex patterns.
        
        For PDF/DOCX: Detects headers and sections using multiple regex patterns
        For CSV/Excel: Treats content as table
        For TXT/MD: Attempts header detection, falls back to single section
        
        Returns
        -------
        List of section dicts:
        [
            {"type": "section", "title": "...", "content": "...", "level": 1},
            {"type": "table", "content": "...", "description": "..."},
            ...
        ]
        """
        sections = self.section_parser.parse_sections(content, doc_type)
        if not sections:
            sections = [{"type": "section", "title": "", "content": content, "level": 0}]
        
        return sections

    def _chunk_section(
        self,
        section: Dict[str, Any],
        document: Dict[str, Any],
        section_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Split a section into chunks based on token size.
        
        Splits on paragraph boundaries (double newlines) to maintain
        semantic coherence within chunks.
        """
        paragraphs = section.get("content", "").split("\n\n")
        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = len(self.tokenizer.encode(para))
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunks.append(
                    self._create_chunk(current_chunk, document, section, section_idx, len(chunks))
                )
                current_chunk = para
                current_tokens = para_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                    current_tokens += para_tokens + 2 
                else:
                    current_chunk = para
                    current_tokens = para_tokens

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
        """
        Create a chunk dictionary with all required metadata.
        """
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
        self,
        section: Dict[str, Any],
        document: Dict[str, Any],
        section_idx: int
    ) -> Dict[str, Any]:
        """
        Create a chunk for table content.
        
        Tables are kept as single chunks to preserve structure.
        """
        chunk = self._create_chunk(
            section.get("content", ""),
            document,
            section,
            section_idx,
            0
        )
        chunk["metadata"]["is_table"] = True
        if "description" in section:
            chunk["metadata"]["table_description"] = section["description"]
        
        return chunk

    @staticmethod
    def _add_relationships(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add parent-child relationships between chunks.
        
        First chunk of each section is parent of subsequent chunks in that section.
        """
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