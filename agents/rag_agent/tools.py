"""
BUG-15 FIX: Thin wrapper that re-exports the canonical tool functions
from `tools/rag_tools.py`.  Agent code can import from this module
for convenience; the ToolRegistry discovers from `tools/rag_tools.py`.
"""

from tools.rag_tools import (
    bm25_search,
    hybrid_search,
    rerank_chunks,
    reciprocal_rank_fusion,
    vector_search,
)

__all__ = [
    "vector_search",
    "hybrid_search",
    "bm25_search",
    "rerank_chunks",
    "reciprocal_rank_fusion",
]
