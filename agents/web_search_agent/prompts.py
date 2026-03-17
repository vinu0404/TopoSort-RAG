"""
Web search agent prompts — production-quality, structured prompts
for Tavily-powered search strategy and synthesis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from security.sanitization import sanitize_user_input, sanitize_web_content
from security.delimiters import wrap_task, wrap_entities, wrap_conversation_history, DELIMITER_SYSTEM_PROMPT
from utils.prompt_utils import format_user_profile


class WebSearchPrompts:

    @staticmethod
    def search_strategy_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:400]}\n"

        entity_str = ", ".join(
            f"{sanitize_user_input(str(k)).text}={sanitize_user_input(str(v)).text}"
            for k, v in entities.items()
        ) if entities else "none"
        long_term_memory_str=" \n\n### User Profile\n" + format_user_profile(long_term_memory or {}) if long_term_memory else ""

        conv_section = ""
        if conversation_history:
            conv_lines = ""
            for turn in (conversation_history[-10:] if isinstance(conversation_history, list) else []):
                role = turn.get("role", "user") if isinstance(turn, dict) else "user"
                content = str(turn.get("content", "") if isinstance(turn, dict) else turn)[:800]
                conv_lines += f"  {role}: {content}\n"
            conv_section = "\n\n### Conversation History\n" + wrap_conversation_history(conv_lines)

        safe_task = sanitize_user_input(task).text

        return f"""{DELIMITER_SYSTEM_PROMPT}

You are the Web Search Strategist for a multi-agent RAG system.

### Current Date
{datetime.now(timezone.utc).strftime('%A, %B %d, %Y')}

### Your Task
Decide the best search strategy and generate an optimised search query for the task below.Use the context from other agents, user profile, and conversation history to inform your decision.

{wrap_task(safe_task)}

{wrap_entities(entity_str)}
{dep_context}
{long_term_memory_str}
{conv_section}
### Available Search Types
1. **basic**  — Fast general web search (default for most queries)
2. **news**   — Restrict to recent news articles (use for current events, recent announcements)
3. **deep**   — Advanced extraction with raw page content (use for technical/research topics needing depth)

### Instructions
- Craft a precise, keyword-rich search query — avoid vague or overly long queries.
- Choose the search type that best fits the information need.
- Set `follow_up_urls` to true ONLY when the snippets alone won't contain enough detail
  (e.g. technical docs, full reports, data tables).

### Output (JSON)
Return EXACTLY this JSON structure, nothing else:
{{
    "search_query": "<optimised search query>",
    "search_type": "basic | news | deep",
    "follow_up_urls": true | false,
    "reasoning": "<one sentence explaining your choice>"
}}"""

    @staticmethod
    def synthesis_prompt(
        task: str,
        tavily_answer: str | None,
        search_results: List[Dict[str, Any]],
        fetched_pages: List[Dict[str, Any]] | None = None,
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
    ) -> str:
        # Format search results — sanitize web content
        results_text = ""
        for i, r in enumerate(search_results[:10], 1):
            title = sanitize_user_input(r.get("title", "Untitled")).text
            snippet = sanitize_web_content(
                r.get("snippet", "")[:400],
                url=r.get("url", ""),
            ).text
            results_text += f"\n[{i}] **{title}**\n"
            results_text += f"    URL: {r.get('url', '')}\n"
            results_text += f"    Snippet: {snippet}\n"

        # Format fetched pages — sanitize web content
        pages_text = ""
        if fetched_pages:
            for i, p in enumerate(fetched_pages, 1):
                page_content = sanitize_web_content(
                    p.get("content", "")[:3000],
                    url=p.get("url", ""),
                ).text
                pages_text += f"\n--- Fetched Page {i}: {p.get('url', '')} ---\n"
                pages_text += f"{page_content}\n"

        # Tavily's own answer
        tavily_section = ""
        if tavily_answer:
            tavily_section = f"\n### Tavily Quick Answer\n{tavily_answer}\n"

        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:400]}\n"

        profile_section = format_user_profile(long_term_memory or {})

        conv_section = ""
        if conversation_history:
            conv_lines = ""
            for turn in (conversation_history[-10:] if isinstance(conversation_history, list) else []):
                role = turn.get("role", "user") if isinstance(turn, dict) else "user"
                content = str(turn.get("content", "") if isinstance(turn, dict) else turn)[:800]
                conv_lines += f"  {role}: {content}\n"
            conv_section = "\n\n### Conversation History\n" + wrap_conversation_history(conv_lines)

        safe_task = sanitize_user_input(task).text

        return f"""{DELIMITER_SYSTEM_PROMPT}

You are a Research Synthesis Expert in a multi-agent RAG system.Your job is to answer the user's question by synthesising information from the web search results and any fetched page content. Use the context from other agents, user profile, and conversation history to inform your answer.

{wrap_task(safe_task)}
{dep_context}
{tavily_section}
{profile_section}
{conv_section}
### Search Results
{results_text}
{pages_text}

### Instructions
1. **Answer the task** using information from the search results and fetched pages above and using the context provided. Be specific and detailed in your answer.
2. **Cite sources** using inline references like [1], [2] matching the result numbers.
3. **Be specific** — include dates, numbers, names, and direct quotes when available.
4. **Acknowledge gaps** — if the search results don't fully answer the question, say so.
5. **Be concise but complete** — aim for a well-structured answer, not a wall of text.
6. **Prefer recent information** over older results when relevance is similar.
7. **Personalise**: Match the user's preferred tone and detail level from the User Profile.
   Respond in the user's preferred language if specified.

### Answer"""

    @staticmethod
    def search_query_prompt(task: str, entities: Dict[str, Any]) -> str:
        safe_task = sanitize_user_input(task).text
        entity_str = ", ".join(
            f"{sanitize_user_input(str(k)).text}={sanitize_user_input(str(v)).text}"
            for k, v in entities.items()
        ) if entities else "none"
        return f"""Generate a concise, keyword-rich web search query for the following task.
Return ONLY the search query string, nothing else.

Task: {safe_task}
Entities: {entity_str}
"""
