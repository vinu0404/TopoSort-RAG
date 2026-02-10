"""
RAG agent prompts — prompts for document retrieval,
reranking, and synthesis with proper citation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from utils.prompt_utils import format_user_profile


class RAGPrompts:

    @staticmethod
    def rerank_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
        return f"""You are a Document Relevance Expert. Your job is to rank document chunks
by how well they answer the user's query.

### User Query
{query}

### Document Chunks
{RAGPrompts._format_chunks(chunks)}

### Instructions
1. Evaluate each chunk for:
   - **Direct relevance**: Does it directly answer or address the query?
   - **Information quality**: Does it contain specific facts, numbers, or actionable info?
   - **Context completeness**: Does it provide enough context to be useful on its own?
2. Rank the MOST relevant chunks first. Return the top 8 maximum.
3. Exclude chunks that are completely irrelevant or contain only boilerplate text.

### Output (JSON only)
Return EXACTLY: {{"ranked_indices": [3, 7, 1, ...]}}

Only include indices of chunks that are actually relevant. If fewer than 8 are relevant,
return fewer.
"""

    @staticmethod
    def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")[:300]
            meta = chunk.get("metadata", {})
            source = meta.get("filename", "unknown")
            section = meta.get("section_title", "")
            header = f"[{idx}] (source: {source}"
            if section:
                header += f", section: {section}"
            header += ")"
            lines.append(f"{header}\n{text}...")
        return "\n\n".join(lines)

    @staticmethod
    def synthesis_prompt(
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_context: str = "",
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        context = ""
        for i, c in enumerate(chunks):
            meta = c.get("metadata", {})
            source = meta.get("filename", "unknown")
            section = meta.get("section_title", "")
            source_label = source
            if section:
                source_label += f" > {section}"
            context += f"\n[Source {i+1}: {source_label}]\n{c.get('text', '')}\n"

        conv_section = ""
        if conversation_context:
            conv_section = f"\n### Conversation History\n{conversation_context}\n"

        profile_section = format_user_profile(long_term_memory or {}, header="User Profile (personalise your output)")

        return f"""You are a Knowledge Synthesis Expert in a multi-agent RAG system.
Your job is to answer the user's question using ONLY the provided document sources.

### User Query
{query}
{conv_section}
{profile_section}
### Document Sources
{context}

### Instructions
1. **Answer the question directly** — start with the key answer, then elaborate.
2. **Cite every claim** using inline references like [1], [2] matching the Source numbers above.
3. **Be specific** — include exact numbers, dates, percentages, names from the documents.
4. **Synthesise across sources** — combine information when multiple sources are relevant.
5. **Acknowledge limitations** — if the documents don't contain enough information to fully
   answer the question, say so explicitly. Do NOT make up information.
6. **Structure your answer** — use paragraphs, bullet points, or numbered lists as appropriate
   for readability.
7. **Quote directly** when the exact wording matters (e.g. policy language, definitions).
8. **Personalise**: Match the user's preferred tone and detail level from the User Profile.
   If the user prefers concise answers, keep it tight. If detailed, be thorough.
   Respond in the user's preferred language if specified.

### Answer"""

    @staticmethod
    def query_expansion_prompt(query: str, entities: Dict[str, Any]) -> str:
        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"
        return f"""You are a Query Expansion Expert. Generate 2-3 alternative search queries
that capture different aspects of the user's information need.

### Original Query
{query}

### Extracted Entities
{entity_str}

### Instructions
- Each alternative should approach the topic from a different angle.
- Include synonyms, related terms, or more specific/general versions.
- Keep each query concise (under 15 words).

### Output (JSON)
Return: {{"queries": ["alt query 1", "alt query 2", "alt query 3"]}}"""
