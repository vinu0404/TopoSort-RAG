"""
RAG Agent – retrieves information from the user's document collection
using hybrid search (dense + BM25) and LLM-based reranking.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.rag_agent.prompts import RAGPrompts
from config.settings import config
from utils.schemas import AgentInput, AgentOutput
import logging
logger = logging.getLogger("rag_agent")

class RAGAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("rag_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = RAGPrompts()

    def get_required_tools(self) -> List[str]:
        return ["vector_search", "hybrid_search", "rerank_chunks"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start_time = time.perf_counter()
        tokens_used = 0
        chunks: List[Dict[str, Any]] = []
        logger.info(f"[RAGAgent] Input: {task_config}")
        try:
            hybrid_search = self.get_tool("hybrid_search")
            rerank = self.get_tool("rerank_chunks")

            original_query = task_config.task
            entities = task_config.entities
            user_id = task_config.metadata.get("user_id", "default")
            enhanced_query = await self._expand_query(original_query, entities)
            tokens_used += 100  
            
            logger.info(f"[RAGAgent] Original query: {original_query}")
            logger.info(f"[RAGAgent] Enhanced query: {enhanced_query}")

            filters = self._build_filters(entities)
            chunks = await hybrid_search(
                query=enhanced_query,
                user_id=user_id,
                filters=filters,
                top_k=20,
            )

            reranked_chunks = await rerank(query=original_query, chunks=chunks, top_k=8)
            sources = self._extract_sources(reranked_chunks)

            logger.info(f"[RAGAgent] Output: {reranked_chunks}")
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=True,
                data={
                    "chunks": reranked_chunks,
                    "sources": sources,
                    "query": original_query
                },
                confidence_score=self._calculate_confidence(reranked_chunks),
                execution_metadata={"attempt_number": 1, "warnings": []},
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start_time) * 1000),
                    "tokens_used": tokens_used,
                    "api_calls_made": 3,  
                    "cache_hits": 0,
                    "estimated_cost": tokens_used * 0.00001,
                },
                depends_on=list(task_config.dependency_outputs.keys()),
                metadata={"sources": sources, "model_used": config.rag_model},
            )

        except asyncio.TimeoutError:
            logger.warning(f"[RAGAgent] Timeout for input: {task_config}")
            return self._create_partial_response(task_config, chunks)

        except Exception:
            logger.exception(f"[RAGAgent] Error for input: {task_config}")
            raise  

    async def _expand_query(self, query: str, entities: Dict[str, Any]) -> str:
        """
        Expand the original query using LLM to generate a more comprehensive search query.
        
        Args:
            query: Original user query
            entities: Extracted entities from the query
            
        Returns:
            Enhanced query string for better retrieval
        """
        
        try:
            prompt = self.prompts.query_expansion_prompt(query, entities)
            
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,  
                max_tokens=150,   
            )
            
            response_text = response.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            enhanced_query = result.get("queries", query)
            
            if not enhanced_query or not enhanced_query.strip():
                logger.warning(f"[RAGAgent] Query expansion returned empty, using original query")
                return query
                
            return enhanced_query
            
        except json.JSONDecodeError as e:
            logger.warning(f"[RAGAgent] Failed to parse query expansion JSON: {e}. Using original query.")
            return query
        except Exception as e:
            logger.warning(f"[RAGAgent] Query expansion failed: {e}. Using original query.")
            return query

    def _build_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if entities.get("date_range"):
            filters["metadata.date"] = entities["date_range"]
        if entities.get("doc_type"):
            filters["metadata.doc_type"] = entities["doc_type"]
        if entities.get("metric"):
            filters["metadata.topic"] = entities["metric"]
        return filters

    @staticmethod
    def _extract_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key = meta.get("filename", "")
            if key and key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "type": "document",
                        "source": key,
                        "page": meta.get("page"),
                        "section": meta.get("section_title"),
                    }
                )
        return sources

    @staticmethod
    def _calculate_confidence(chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        scores = [c.get("score", 0.0) for c in chunks]
        return min(1.0, sum(scores) / max(len(scores), 1))

    def _create_partial_response(
        self, task_config: AgentInput, chunks: List[Dict[str, Any]]
    ) -> AgentOutput:
        return AgentOutput(
            agent_id=task_config.agent_id,
            agent_name=self.agent_name,
            task_description=task_config.task,
            task_done=False,
            partial_data={
                "documents_found": len(chunks),
                "results": chunks[:3] if chunks else [],
                "search_completed": False,
            },
            error="timeout",
            confidence_score=0.0,
            depends_on=list(task_config.dependency_outputs.keys()),
        )