"""
Composer Agent — synthesises outputs from all agents into a final
answer with inline citations, supporting both sync and streaming.
"""

from __future__ import annotations

import logging
from typing import  AsyncIterator, List

from config.settings import config
from utils.llm_providers import BaseLLMProvider
from utils.schemas import (
    AgentOutput,
    ComposerInput,
    ComposerOutput,
    Source,
)

logger = logging.getLogger(__name__)


class ComposerAgent:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    async def compose(self, composer_input: ComposerInput) -> ComposerOutput:
        prompt = self._build_prompt(composer_input)
        logger.info(f"[ComposerAgent] Input: {composer_input}")
        logger.debug(f"[ComposerAgent] Prompt: {prompt[:500]}")
        response = await self.llm.generate(
            prompt=prompt,
            temperature=config.composer_temperature,
            model=config.composer_model,
        )
        answer_text = response if isinstance(response, str) else str(response)
        answer_text = self._append_source_list(answer_text, composer_input.all_sources)

        logger.info(f"[ComposerAgent] Output: {answer_text[:500]}")
        return ComposerOutput(
            query_id=composer_input.query_id,
            answer=answer_text,
            sources=composer_input.all_sources,
        )


    async def stream(self, composer_input: ComposerInput) -> AsyncIterator[str]:
        """Yield text chunks for SSE consumption."""
        prompt = self._build_prompt(composer_input)

        async for chunk in self.llm.stream(
            prompt=prompt,
            temperature=config.composer_temperature,
            model=config.composer_model,
        ):
            yield chunk

        yield "\n\nSources:\n"
        for idx, source in enumerate(composer_input.all_sources, start=1):
            yield f"[{idx}] {self._format_source(source)}\n"


    def _build_prompt(self, ci: ComposerInput) -> str:
        successful = [r for r in ci.agent_results if r.task_done]
        failed = [r for r in ci.agent_results if not r.task_done]
        partial = [r for r in failed if r.partial_data is not None]
        completely_failed = [r.agent_name for r in failed if r.partial_data is None]

        return f"""You are the Final Response Composer for a multi-agent RAG system.
Your job is to synthesise outputs from specialised agents into one coherent, polished answer.

### Original User Query
{ci.original_query}

### User Profile
{ci.long_term_memory.model_dump() if hasattr(ci.long_term_memory, 'model_dump') else ci.long_term_memory}

### Conversation History
{self._format_conversation_history(ci.conversation_history)}

### Successful Agent Outputs
{self._format_agent_outputs(successful)}

### Partial Data (from agents that partially completed)
{self._format_agent_outputs(partial)}

### Failed Agents (could not produce results)
{completely_failed if completely_failed else "(none)"}

### Available Sources
{self._format_sources_for_prompt(ci.all_sources)}

### Composition Instructions
1. **Start with the answer** — don't repeat the question or say "Based on the analysis...".
   Get straight to the point.
2. **Synthesise across agents** — weave information from different agents into a unified
   narrative. Don't just concatenate their outputs.
3. **Cite sources** using [1], [2], [3] inline references matching the source list above.
4. **Be specific** — include exact numbers, dates, names, and direct quotes from the data.
5. **Structure for readability**:
   - Use paragraphs for narrative answers.
   - Use bullet points or numbered lists for multiple items.
   - Use bold for key terms or findings.
6. **Tone**: {ci.long_term_memory.preferences.tone if hasattr(ci.long_term_memory, 'preferences') else 'professional'}
7. **Detail level**: {ci.long_term_memory.preferences.detail_level if hasattr(ci.long_term_memory, 'preferences') else 'balanced'}
8. **Handle failures gracefully** — if some agents failed, acknowledge what information
   is unavailable without being apologetic. Offer the best answer possible.
9. **Match conversational flow** — if there's conversation history, maintain continuity
   and refer back to earlier points when relevant.
10. If an agent found emails, web results, or code output, present them in a
    user-friendly format (not raw JSON).
11. **Address the user by name** if their name is available in the User Profile.
12. **Language**: If the user profile specifies a preferred language, write the entire
    response in that language.

### Answer"""


    @staticmethod
    def _format_agent_outputs(outputs: List[AgentOutput]) -> str:
        if not outputs:
            return "(none)"
        lines = []
        for o in outputs:
            data = o.data if o.task_done else o.partial_data
            lines.append(
                f"  [{o.agent_id} / {o.agent_name}]\n"
                f"    Task: {o.task_description}\n"
                f"    Data: {str(data)[:1000]}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_conversation_history(turns) -> str:
        if not turns:
            return "(no prior conversation)"
        lines = []
        for t in turns[-6:]:
            if hasattr(t, "user_query"):
                lines.append(f"  User: {t.user_query[:300]}")
                lines.append(f"  Assistant: {t.composer_answer[:300]}")
            elif isinstance(t, dict):
                lines.append(f"  {t.get('role', 'user')}: {str(t.get('content', ''))[:300]}")
        return "\n".join(lines) if lines else "(no prior conversation)"

    @staticmethod
    def _format_sources_for_prompt(sources: List[Source]) -> str:
        if not sources:
            return "(none)"
        return "\n".join(
            f"  [{i}] {s.source} ({s.type})"
            for i, s in enumerate(sources, start=1)
        )

    @staticmethod
    def _format_source(source: Source) -> str:
        if source.type == "document":
            return f"{source.source}, page {source.page}" if source.page else source.source
        if source.type == "web":
            return f"{source.source} ({source.url})" if source.url else source.source
        return source.source

    @staticmethod
    def _append_source_list(text: str, sources: List[Source]) -> str:
        if not sources:
            return text
        text += "\n\nSources:\n"
        for idx, s in enumerate(sources, start=1):
            text += f"[{idx}] {ComposerAgent._format_source(s)}\n"
        return text
