"""
Web Search Agent — real-time web search via Tavily API
"""

from __future__ import annotations

import time
from typing import Dict, List

from agents.base_agent import BaseAgent
from agents.web_search_agent.prompts import WebSearchPrompts
from config.settings import config
from utils.schemas import AgentInput, AgentOutput

import logging
logger = logging.getLogger("web_search_agent")

class WebSearchAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("web_search_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = WebSearchPrompts()

    def get_required_tools(self) -> List[str]:
        return ["web_search", "fetch_url"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()

        logger.info(f"[WebSearchAgent] Input: {task_config}")
        try:
            search = self.get_tool("web_search")
            fetch = self.get_tool("fetch_url")
            search_news = self.get_tool("web_search_news")
            search_deep = self.get_tool("web_search_deep")

            effective_task = await self._effective_task(task_config)

            strategy_prompt = self.prompts.search_strategy_prompt(
                task=effective_task,
                entities=task_config.entities,
                dependency_outputs=task_config.dependency_outputs,
                long_term_memory=task_config.long_term_memory,
                conversation_history=task_config.conversation_history,
            )
            logger.info(f"========================================")
            logger.info(f"[WebSearchAgent] conversation_history: {task_config.conversation_history}")

            logger.info(f"========================================")
            strategy: str = await self.llm.generate(
                prompt=strategy_prompt,
                temperature=config.web_temperature,
                model=config.web_model,
                output_schema={
                    "search_query": "string",
                    "search_type": "basic | news | deep",
                    "follow_up_urls": "boolean",
                    "reasoning": "string",
                },
            )
            if isinstance(strategy, dict):
                search_query = strategy.get("search_query", task_config.task)
                search_type = strategy.get("search_type", "basic")
                should_follow_up = strategy.get("follow_up_urls", False)
            else:
                search_query = str(strategy).strip().strip('"')
                search_type = "basic"
                should_follow_up = True
            if search_type == "news" and search_news:
                search_result = await search_news(query=search_query, num_results=7)
            elif search_type == "deep" and search_deep:
                search_result = await search_deep(query=search_query, num_results=5)
            else:
                search_result = await search(
                    query=search_query,
                    num_results=7,
                    search_depth="basic",
                    include_answer=True,
                )

            tavily_answer = search_result.get("answer")
            results = search_result.get("results", [])
            fetched_contents = []
            if should_follow_up and results and fetch:
                top_urls = [r["url"] for r in results[:2] if r.get("url")]
                for url in top_urls:
                    content = await fetch(url=url)
                    if content and not content.get("error"):
                        fetched_contents.append(content)
            synthesis_prompt = self.prompts.synthesis_prompt(
                task=effective_task,
                tavily_answer=tavily_answer,
                search_results=results,
                fetched_pages=fetched_contents,
                dependency_outputs=task_config.dependency_outputs,
                long_term_memory=task_config.long_term_memory,
                conversation_history=task_config.conversation_history,
            )
            final_answer: str = await self.llm.generate(
                prompt=synthesis_prompt,
                temperature=config.web_temperature,
                model=config.web_model,
            )

            sources = [
                {
                    "type": "web",
                    "source": r.get("title", ""),
                    "url": r.get("url", ""),
                    "excerpt": r.get("snippet", "")[:300],
                    "relevance_score": r.get("score", 0.0),
                }
                for r in results
            ]

            confidence = self._compute_confidence(results, tavily_answer)

            logger.info(f"[WebSearchAgent] Output: {final_answer}")
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=effective_task,
                task_done=True,
                result=final_answer,
                data={
                    "search_query": search_query,
                    "search_type": search_type,
                    "tavily_answer": tavily_answer,
                    "search_results": results,
                    "fetched_pages": len(fetched_contents),
                    "sources": sources,
                },
                confidence_score=confidence,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                    "api_calls_made": 1 + len(fetched_contents) + 1, 
                },
                depends_on=list(task_config.dependency_outputs.keys()),
                metadata={"sources": sources},
            )

        except Exception:
            logger.exception(f"[WebSearchAgent] Error for input: {task_config}")
            raise


    @staticmethod
    def _compute_confidence(results: List[Dict], tavily_answer: str | None) -> float:
        if not results:
            return 0.1
        scores = [r.get("score", 0.0) for r in results if r.get("score")]
        avg_score = sum(scores) / max(len(scores), 1) if scores else 0.3
        if tavily_answer:
            avg_score = min(1.0, avg_score + 0.15)
        return round(min(1.0, avg_score), 2)
