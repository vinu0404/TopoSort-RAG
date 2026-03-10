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
    PersonaContext,
    Source,
)

logger = logging.getLogger(__name__)


class ComposerAgent:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider
        self.last_stream_usage: dict = {}

    async def compose(self, composer_input: ComposerInput) -> ComposerOutput:
        prompt = self._build_prompt(composer_input)
        logger.info(f"[ComposerAgent] Input: {composer_input}")
        logger.debug(f"[ComposerAgent] Prompt: {prompt[:500]}")
        result = await self.llm.generate(
            prompt=prompt,
            temperature=config.composer_temperature,
            model=config.composer_model,
        )
        answer_text = result.text
        answer_text = self._append_source_list(answer_text, composer_input.all_sources)

        logger.info(f"[ComposerAgent] Output: {answer_text[:500]}")
        return ComposerOutput(
            query_id=composer_input.query_id,
            answer=answer_text,
            sources=composer_input.all_sources,
            usage=result.usage,
        )


    async def stream(self, composer_input: ComposerInput) -> AsyncIterator[str]:
        """Yield text chunks for SSE consumption.  Token usage is stored
        in ``self.last_stream_usage`` after iteration completes."""
        prompt = self._build_prompt(composer_input)
        self.last_stream_usage = {}

        sr = await self.llm.stream_with_usage(
            prompt=prompt,
            temperature=config.composer_temperature,
            model=config.composer_model,
        )
        async for chunk in sr:
            yield chunk
        self.last_stream_usage = sr.usage

        yield "\n\nSources:\n"
        for idx, source in enumerate(composer_input.all_sources, start=1):
            yield f"[{idx}] {self._format_source(source)}\n"


    def _build_prompt(self, ci: ComposerInput) -> str:
        successful = [r for r in ci.agent_results if r.task_done]
        failed = [r for r in ci.agent_results if not r.task_done]
        partial = [r for r in failed if r.partial_data is not None]
        completely_failed = [r.agent_name for r in failed if r.partial_data is None]

        # Detect HITL-overridden tasks
        hitl_note = self._detect_hitl_overrides(ci.agent_results)

        # Detect conversation-memory queries (no agents dispatched)
        is_memory_query = not successful and not partial and not completely_failed

        if is_memory_query:
            return f"""You are {config.bot_name}, the Final Response Composer for a multi-agent RAG system.
When the user asks who you are, your name, or what you are, always say you are {config.bot_name}.with the following persona:{self._format_persona_block(ci.persona)} tuned to answer questions about the conversation history and user profile.
{self._format_persona_block(ci.persona)}The user has asked a question about the conversation itself — no specialised agents were needed.
Answer ENTIRELY from the conversation history and user profile below.

### Original User Query
{ci.original_query}

### User Profile
{ci.long_term_memory.model_dump() if hasattr(ci.long_term_memory, 'model_dump') else ci.long_term_memory}

### Conversation History
{self._format_conversation_history(ci.conversation_history)}

### Instructions for Conversation-Memory Queries
1. **Answer directly** from the conversation history above. Do NOT say "I don't have
   access to that information" — the history IS your information source.
2. If the user asks "what did I ask?", "what were my last questions?", "summarise our
   chat", etc., list their previous queries clearly with any relevant answers.
3. If the user asks "what topics have we discussed?", extract the key themes/topics
   from the conversation history.
4. Be precise — quote exact queries and answers when the user asks for them.
5. Use a natural, conversational tone. Don't over-explain.
6. **Address the user by name** if their name is available in the User Profile.
7. **Language**: If the user profile specifies a preferred language, respond in that language.

### Answer"""

        return f"""You are {config.bot_name}, the Final Response Composer for a multi-agent RAG system.
When the user asks who you are, your name, or what you are, always say you are {config.bot_name}.
{self._format_persona_block(ci.persona)}Your job is to synthesise outputs from specialised agents into one coherent, polished answer.

### Original User Query
{ci.original_query}

### User Profile
{ci.long_term_memory.model_dump() if hasattr(ci.long_term_memory, 'model_dump') else ci.long_term_memory}

### Conversation History
{self._format_conversation_history(ci.conversation_history)}

### Successful Agent Outputs
{self._format_agent_outputs(successful)}
{hitl_note}
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
    def _format_persona_block(persona: PersonaContext | None) -> str:
        """Build a persona instruction block for the system prompt."""
        if not persona:
            return ""
        return (
            f"### Persona\n"
            f"You MUST adopt the following persona for your entire response.\n"
            f"Name: {persona.name}\n"
            f"Description: {persona.description}\n"
            f"Stay fully in character — tone, vocabulary, and personality must match this persona.\n\n"
        )

    @staticmethod
    def _detect_hitl_overrides(outputs: list[AgentOutput]) -> str:
        """
        Detect whether any agent had its task enhanced or overridden via HITL,
        and tell the Composer how to present the results.
        """
        notices = []
        for o in outputs:
            if not o.task_description:
                continue
            if "User Override Instructions" in o.task_description:
                notices.append(
                    f"  - **{o.agent_name}** was OVERRIDDEN by the user during HITL approval. "
                    f"Present the data this agent actually returned, NOT the original query topic."
                )
            elif "User Enhanced Instructions" in o.task_description:
                notices.append(
                    f"  - **{o.agent_name}** was ENHANCED by the user during HITL approval with additional scope. "
                    f"Include BOTH the original AND enhanced data in your answer.For better personalisation, prioritise the ENHANCED data but use the original data as well if relevant."
                )
        if notices:
            return (
                "\n### \u26a0 HITL Instruction Notice\n"
                "The user modified the task for the following agent(s) during approval:\n"
                + "\n".join(notices)
                + "\nAnswer based on the DATA the agent(s) actually returned,with composing the answer in a way that respects the user's modifications of instructions,"
                "combining or replacing as indicated above.\n\n"
            )
        return "\n"

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

    # ── Voice summary (runs in parallel with text composer) ────────────

    async def generate_voice_summary_from_input(self, ci: ComposerInput) -> str:
        """
        Generate a concise spoken summary directly from the raw agent data.

        This receives the same ComposerInput as the text composer so it can
        run in parallel — it does NOT depend on the finished text answer.
        """
        successful = [r for r in ci.agent_results if r.task_done]
        partial = [r for r in ci.agent_results if not r.task_done and r.partial_data]
        is_memory_query = not successful and not partial

        # Build data context
        if is_memory_query:
            data_block = (
                f"Conversation History:\n"
                f"{self._format_conversation_history(ci.conversation_history)}"
            )
        else:
            data_block = self._format_agent_outputs(successful)
            if partial:
                data_block += f"\nPartial data:\n{self._format_agent_outputs(partial)}"

        persona_block = ""
        if ci.persona:
            persona_block = (
                f"Persona: {ci.persona.name}\n"
                f"Speak in this style: {ci.persona.description}\n"
                f"Keep the persona's tone and personality in the spoken summary.\n\n"
            )

        prompt = (
            f"You are a voice assistant. Answer the user's question as a concise "
            f"spoken response suitable for text-to-speech.\n\n"
            f"{persona_block}"
            f"User question: {ci.original_query}\n\n"
            f"Data from agents:\n{data_block[:3000]}\n\n"
            f"Rules:\n"
            f"- Use natural, conversational spoken language.\n"
            f"- NO markdown, NO bold, NO headers, NO bullet points.\n"
            f"- NO citation numbers like [1], [2].\n"
            f"- NO code blocks or raw data tables.\n"
            f"- Keep the key facts and conclusions.\n"
            f"- Use short, clear sentences.\n"
            f"- Maximum 3-4 sentences.\n\n"
            f"Spoken answer:"
        )

        result = await self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            model=config.composer_model,
        )
        return result.text.strip()
