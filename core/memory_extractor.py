"""
Memory Extractor — LLM layer that inspects every user query for long term memory :personal
information (critical facts / preferences) and persists any findings to
long-term memory.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import config
from core.memory_manager import MemoryManager
from utils.llm_providers import BaseLLMProvider, get_llm_provider
from utils.schemas import CriticalFacts, LongTermMemory, UserPreferences

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are a Memory Extraction Agent.
Analyse the user's message and extract any NEW personal information that should
be remembered for future conversations.

### Categories to extract

**critical_facts** (factual info about the user):
- user_name: the user's name
- company_name: where they work
- job_title: their role / position
- recent_projects: project names they mention

**preferences** (how they want to be addressed / served):
- tone: "professional" | "casual" | "formal" | "friendly"
- detail_level: "concise" | "detailed" | "technical"
- language: ISO-639-1 code, e.g. "en", "es", "fr"

### Rules
1. Only extract facts that are **explicitly stated** — never infer.
2. **Compare against Current Memory below.** If the user restates something
   already stored with the SAME value, do NOT include it. Only return fields
   whose value is genuinely NEW or CHANGED.
3. For `recent_projects`: only include project names that are NOT already in
   the list. If the user mentions a project that's already stored, skip it.
4. If the message contains NO new or changed data at all, return `{{"found": false}}`.
5. If NEW/CHANGED facts ARE found, return
   `{{"found": true, "critical": {{...}}, "preferences": {{...}}}}`.
   Only include keys whose values differ from Current Memory.
6. Return valid JSON only — no markdown, no explanation.

### User Message
{query}

### Current Memory
{current_memory}

### Output (JSON)
"""


class MemoryExtractor:
    """
    Stateless LLM wrapper that extracts user facts from a query string.

    """

    def __init__(self, llm_provider: BaseLLMProvider | None = None):
        self.llm = llm_provider or get_llm_provider(
            config.master_model_provider,
            default_model=config.master_model,
        )

    async def extract(
        self,
        query: str,
        current_memory: LongTermMemory | None = None,
    ) -> Dict[str, Any] | None:
        """
        Run the extraction LLM call.

        Returns the parsed dict with ``found``, ``critical``, ``preferences``
        keys, or ``None`` if nothing was found / parsing failed.
        """
        mem_str = ""
        if current_memory:
            mem_str = json.dumps(current_memory.model_dump(), indent=2)
        else:
            mem_str = "(no existing memory)"

        prompt = _EXTRACTION_PROMPT.format(query=query, current_memory=mem_str)

        try:
            raw = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,  
                model=config.master_model,
                max_tokens=512,    
            )

            if isinstance(raw, str):
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
                parsed = json.loads(cleaned)
            elif isinstance(raw, dict):
                parsed = raw
            else:
                return None

            if not parsed.get("found", False):
                return None

            return parsed

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Memory extraction failed: %s", exc)
            return None

    async def extract_and_store(
        self,
        query: str,
        user_id: str,
        memory_manager: MemoryManager,
        db_session: AsyncSession | None = None,
    ) -> None:
        """get current long term memory and run extraction → diff → persist only new/changed fields
        """
        try:
            current = await memory_manager.get_long_term_memory(
                user_id, db_session=db_session,
            )
            extracted = await self.extract(query, current_memory=current)

            if extracted is None:
                logger.debug(
                    "No new personal data found in query for user %s", user_id
                )
                return
            updates: Dict[str, Any] = {}

            if "critical" in extracted:
                critical_diff = self._diff_critical(
                    current.critical, extracted["critical"]
                )
                if critical_diff:
                    updates["critical"] = critical_diff

            if "preferences" in extracted:
                pref_diff = self._diff_preferences(
                    current.preferences, extracted["preferences"]
                )
                if pref_diff:
                    updates["preferences"] = pref_diff

            if not updates:
                logger.debug(
                    "Extracted data is identical to existing memory for user %s — skipping DB write",
                    user_id,
                )
                return

            await memory_manager.update_long_term_memory(
                user_id, updates, db_session=db_session,
            )
            logger.info(
                "Stored NEW memory fields for user %s: %s",
                user_id,
                {k: list(v.keys()) if isinstance(v, dict) else v for k, v in updates.items()},
            )

        except Exception:
            logger.exception(
                "Memory extraction background task failed for user %s", user_id
            )

    @staticmethod
    def _diff_critical(
        existing: "CriticalFacts", extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return only critical_facts fields whose value differs from *existing*."""
        diff: Dict[str, Any] = {}
        for key, new_val in extracted.items():
            if not hasattr(existing, key):
                continue
            old_val = getattr(existing, key)
            if isinstance(old_val, list) and isinstance(new_val, list):
                old_set = set(old_val)
                truly_new = [v for v in new_val if v not in old_set]
                if truly_new:
                    diff[key] = old_val + truly_new
            elif new_val != old_val and new_val is not None:
                diff[key] = new_val

        return diff

    @staticmethod
    def _diff_preferences(
        existing: "UserPreferences", extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return only preference fields whose value differs from *existing*."""
        diff: Dict[str, Any] = {}
        for key, new_val in extracted.items():
            if not hasattr(existing, key):
                continue
            old_val = getattr(existing, key)
            if new_val != old_val and new_val is not None:
                diff[key] = new_val
        return diff
