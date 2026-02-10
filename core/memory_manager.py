"""
Memory Manager — conversation history + long-term memory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import config
from utils.llm_providers import BaseLLMProvider, get_llm_provider
from utils.schemas import (
    AgentConversationHistory,
    ConversationSummary,
    ConversationTurn,
    CriticalFacts,
    LongTermMemory,
    UserPreferences,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages per-session conversation turns and per-user long-term memory.
    """

    _instance: "MemoryManager | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, llm_provider: BaseLLMProvider | None = None):
        if self._initialized:
            return
        self._initialized = True
        self.llm = llm_provider or get_llm_provider(
            config.master_model_provider,
            default_model=config.master_model,
        )
        self._turns: Dict[str, List[ConversationTurn]] = {}
        self._summaries: Dict[str, List[ConversationSummary]] = {}
        self._long_term: Dict[str, LongTermMemory] = {}

    async def add_turn(
        self,
        user_id: str,
        query: str,
        answer: str,
        db_session: AsyncSession | None = None,
        conversation_id: str | None = None,
    ) -> ConversationTurn:
        turns = self._turns.setdefault(user_id, [])
        turn = ConversationTurn(
            turn=len(turns) + 1,
            user_query=query,
            composer_answer=answer,
            timestamp=str(__import__("time").time()),
        )
        turns.append(turn)
        interval = config.conversation_summary_interval
        if len(turns) % interval == 0:
            group = turns[-interval:]
            await self._summarise_group(
                user_id, group,
                db_session=db_session,
                conversation_id=conversation_id,
            )

        return turn

    async def _hydrate_from_db(
        self,
        user_id: str,
        db_session: AsyncSession,
        conversation_id: str,
    ) -> None:
        """
        One-time load of conversation history from PostgreSQL into the
        in-memory cache.  Called only when ``_turns[user_id]`` is empty
        (cold start / process restart).  After hydration the normal
        in-memory path takes over — no repeated DB reads per request.
        """
        from database.helpers import (
            load_conversation_messages,
            load_conversation_summaries,
        )

        # ── Load messages → rebuild _turns ───────────────────────────────
        raw_messages = await load_conversation_messages(
            db_session, conversation_id,
        )
        turns: List[ConversationTurn] = []
        turn_num = 0
        i = 0
        while i < len(raw_messages):
            msg = raw_messages[i]
            if msg["role"] == "user":
                user_q = msg["content"]
                assistant_a = ""
                if i + 1 < len(raw_messages) and raw_messages[i + 1]["role"] == "assistant":
                    assistant_a = raw_messages[i + 1]["content"]
                    i += 1
                turn_num += 1
                turns.append(ConversationTurn(
                    turn=turn_num,
                    user_query=user_q,
                    composer_answer=assistant_a,
                    timestamp="",
                ))
            i += 1

        if turns:
            self._turns[user_id] = turns
            logger.info(
                "Hydrated %d conversation turns from DB for user %s",
                len(turns), user_id,
            )

        # ── Load summaries → rebuild _summaries ──────────────────────────
        raw_summaries = await load_conversation_summaries(
            db_session, conversation_id,
        )
        summaries: List[ConversationSummary] = []
        covered_so_far = 0
        for rs in raw_summaries:
            n = rs["turns_covered"]
            turn_ids = list(range(covered_so_far + 1, covered_so_far + n + 1))
            covered_so_far += n
            summaries.append(ConversationSummary(
                turns=turn_ids,
                summary=rs["summary_text"],
                key_points=[],
                timestamp="",
            ))

        if summaries:
            self._summaries[user_id] = summaries
            logger.info(
                "Hydrated %d conversation summaries from DB for user %s",
                len(summaries), user_id,
            )

    async def get_agent_history(
        self,
        user_id: str,
        *,
        db_session: AsyncSession | None = None,
        conversation_id: str | None = None,
    ) -> AgentConversationHistory:
        """
        Build the history object that every agent receives.

        On first call (cache empty), hydrates from PostgreSQL if a
        ``db_session`` and ``conversation_id`` are provided.
        """
        # Hydrate from DB on cold start
        if user_id not in self._turns and db_session and conversation_id:
            await self._hydrate_from_db(user_id, db_session, conversation_id)

        summaries = self._summaries.get(user_id, [])
        turns = self._turns.get(user_id, [])
        interval = config.conversation_summary_interval
        summarised_count = len(summaries) * interval
        unsummarised = turns[summarised_count:]
        recent_turns: List[ConversationTurn] = []
        if summaries:
            last = summaries[-1]
            recent_turns = [t for t in turns if t.turn in last.turns]

        return AgentConversationHistory(
            summaries=summaries[-config.max_conversation_history:],
            recent_turns=recent_turns,
            recent_unsummarized_turns=unsummarised,
        )

    async def get_conversation_history_for_agents(
        self,
        user_id: str,
        *,
        db_session: AsyncSession | None = None,
        conversation_id: str | None = None,
    ) -> List[Dict[str, str]]:
        """
        Build the flat List[Dict] conversation history that agents receive
        via AgentInput.conversation_history.

        Format: [{"role": "user"|"assistant"|"system", "content": "..."}, ...]

        Includes:
          1. Summaries (as system messages) — compressed older context
          2. Recent turns from the last summary group
          3. Unsummarised turns — most recent context

        Falls back to PostgreSQL on first call if cache is cold.
        """
        agent_history = await self.get_agent_history(
            user_id,
            db_session=db_session,
            conversation_id=conversation_id,
        )
        history: List[Dict[str, str]] = []

        for summary in agent_history.summaries:
            history.append({
                "role": "system",
                "content": f"[Conversation summary] {summary.summary}",
            })

        for turn in agent_history.recent_turns:
            history.append({"role": "user", "content": turn.user_query})
            history.append({"role": "assistant", "content": turn.composer_answer})

        for turn in agent_history.recent_unsummarized_turns:
            history.append({"role": "user", "content": turn.user_query})
            history.append({"role": "assistant", "content": turn.composer_answer})

        return history

    async def get_long_term_memory(
        self, user_id: str, db_session: AsyncSession | None = None
    ) -> LongTermMemory:
        """Load long-term memory from cache, falling back to the database."""
        if user_id in self._long_term:
            return self._long_term[user_id]

        if db_session is not None:
            from database.models import UserLongTermMemory

            result = await db_session.execute(
                select(UserLongTermMemory).where(
                    UserLongTermMemory.user_id == user_id
                )
            )
            row = result.scalar_one_or_none()
            if row is not None:
                mem = LongTermMemory(
                    user_id=user_id,
                    critical=CriticalFacts(**row.critical_facts),
                    preferences=UserPreferences(**row.preferences),
                )
                self._long_term[user_id] = mem
                return mem

        mem = LongTermMemory(
            user_id=user_id,
            critical=CriticalFacts(),
            preferences=UserPreferences(),
        )
        self._long_term[user_id] = mem
        return mem

    async def update_long_term_memory(
        self,
        user_id: str,
        updates: Dict[str, Any],
        db_session: AsyncSession | None = None,
    ) -> None:
        """Apply incremental updates and optionally persist to the database."""
        mem = await self.get_long_term_memory(user_id)
        if "critical" in updates:
            for k, v in updates["critical"].items():
                if hasattr(mem.critical, k):
                    setattr(mem.critical, k, v)
        if "preferences" in updates:
            for k, v in updates["preferences"].items():
                if hasattr(mem.preferences, k):
                    setattr(mem.preferences, k, v)
        self._long_term[user_id] = mem

        if db_session is not None:
            from database.models import UserLongTermMemory

            stmt = pg_insert(UserLongTermMemory).values(
                user_id=user_id,
                critical_facts=mem.critical.model_dump(),
                preferences=mem.preferences.model_dump(),
            ).on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "critical_facts": mem.critical.model_dump(),
                    "preferences": mem.preferences.model_dump(),
                },
            )
            await db_session.execute(stmt)
            logger.info("Persisted long-term memory for user %s", user_id)

    async def _summarise_group(
        self,
        user_id: str,
        group: List[ConversationTurn],
        db_session: AsyncSession | None = None,
        conversation_id: str | None = None,
    ) -> ConversationSummary:
        text = "\n".join(
            f"User: {t.user_query}\nAssistant: {t.composer_answer}" for t in group
        )
        prompt = (
            f"Summarise the following conversation turns in 2-3 sentences. "
            f"Also list any key facts or data points.\n\n{text}\n\n"
            f'Return JSON: {{"summary": "...", "key_points": ["..."]}}'
        )
        result = await self.llm.generate(
            prompt=prompt,
            output_schema={"summary": "string", "key_points": ["string"]},
            temperature=0.2,
        )

        if not isinstance(result, dict):
            result = {"summary": str(result), "key_points": []}

        summary = ConversationSummary(
            turns=[t.turn for t in group],
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            timestamp=str(__import__("time").time()),
        )
        self._summaries.setdefault(user_id, []).append(summary)

        if db_session is not None and conversation_id is not None:
            try:
                from database.helpers import save_conversation_summary

                await save_conversation_summary(
                    db_session,
                    conversation_id,
                    summary_text=summary.summary,
                    turns_covered=len(group),
                )
                logger.info(
                    "Persisted conversation summary for user %s (turns %s)",
                    user_id,
                    summary.turns,
                )
            except Exception:
                logger.exception("Failed to persist conversation summary")

        return summary
