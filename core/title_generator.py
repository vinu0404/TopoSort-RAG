"""
Generate a short conversation title from the first user message via LLM.
"""

from __future__ import annotations

import logging
from utils.llm_providers import BaseLLMProvider

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "Generate a short, descriptive title (max 7 words) for a conversation "
    "that starts with this user message. Return ONLY the title text, nothing else.\n\n"
    "User message: {query}"
)


async def generate_title(llm: BaseLLMProvider, query: str, model: str | None = None) -> str:
    """Return a short LLM-generated title for the given query."""
    try:
        result = await llm.generate(
            prompt=_TITLE_PROMPT.format(query=query[:500]),
            temperature=0.3,
            model=model,
        )
        title = result.text.strip().strip('"').strip("'")
        # Fallback if LLM returns something too long or empty
        if not title or len(title) > 100:
            return query[:80]
        return title
    except Exception:
        logger.warning("Title generation failed, using fallback", exc_info=True)
        return query[:80]
