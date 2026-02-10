"""
Web search agent prompts — production-quality, structured prompts
for Tavily-powered search strategy and synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.prompt_utils import format_user_profile


class WebSearchPrompts:

    @staticmethod
    def search_strategy_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:400]}\n"

        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"

        return f"""You are the Web Search Strategist for a multi-agent RAG system.

### Your Task
Decide the best search strategy and generate an optimised search query for the task below.

### Task
{task}

### Extracted Entities
{entity_str}
{dep_context}

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
    ) -> str:
        # Format search results
        results_text = ""
        for i, r in enumerate(search_results[:10], 1):
            results_text += f"\n[{i}] **{r.get('title', 'Untitled')}**\n"
            results_text += f"    URL: {r.get('url', '')}\n"
            results_text += f"    Snippet: {r.get('snippet', '')[:400]}\n"

        # Format fetched pages
        pages_text = ""
        if fetched_pages:
            for i, p in enumerate(fetched_pages, 1):
                pages_text += f"\n--- Fetched Page {i}: {p.get('url', '')} ---\n"
                pages_text += f"{p.get('content', '')[:3000]}\n"

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

        return f"""You are a Research Synthesis Expert in a multi-agent RAG system.

### Original Task
{task}
{dep_context}
{tavily_section}
{profile_section}
### Search Results
{results_text}
{pages_text}

### Instructions
1. **Answer the task** using information from the search results and fetched pages above.
2. **Cite sources** using inline references like [1], [2] matching the result numbers.
3. **Be specific** — include dates, numbers, names, and direct quotes when available.
4. **Acknowledge gaps** — if the search results don't fully answer the question, say so.
5. **Be concise but complete** — aim for a well-structured answer, not a wall of text.
6. **Prefer recent information** over older results when relevance is similar.
7. **Personalise**: Match the user's preferred tone and detail level from the User Profile.
   Respond in the user's preferred language if specified.

### Answer"""

    # Keep backward compatibility for simpler uses
    @staticmethod
    def search_query_prompt(task: str, entities: Dict[str, Any]) -> str:
        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"
        return f"""Generate a concise, keyword-rich web search query for the following task.
Return ONLY the search query string, nothing else.

Task: {task}
Entities: {entity_str}
"""
