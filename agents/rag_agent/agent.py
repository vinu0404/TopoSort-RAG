"""
RAG Agent – retrieves information from the user's document collection
using a two-level retrieval pipeline:

  Stage 1  →  description-level search → get relevant doc_ids
  Stage 2  →  hybrid (dense ‖ BM25) within matched docs → RRF merge
  Optional →  LLM reranking (controlled by config.rag_use_llm_reranking)
  Final    →  LLM synthesis with citations
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
        return ["two_level_search", "rerank_chunks"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start_time = time.perf_counter()
        tokens_used = 0
        chunks: List[Dict[str, Any]] = []
        logger.info("[RAGAgent] Input: agent_id=%s task=%s tools=%s", task_config.agent_id, task_config.task[:150], task_config.tools)
        try:
            two_level_search = self.get_tool("two_level_search")
            rerank = self.get_tool("rerank_chunks")

            original_query = task_config.task
            entities = task_config.entities
            dependency_outputs = task_config.dependency_outputs
            user_id = task_config.metadata.get("user_id", "default")
            active_web_ids = task_config.metadata.get("active_web_collection_ids", [])
            selected_doc_ids = task_config.metadata.get("selected_doc_ids", [])
            enhanced_query, expand_tokens = await self._expand_query(original_query, entities, dependency_outputs)
            tokens_used += expand_tokens

            logger.info(f"[RAGAgent] Original query: {original_query}")
            logger.info(f"[RAGAgent] Enhanced query: {enhanced_query}")
            filters = self._build_filters(entities)

            # ── Two-level retrieval ─────────────────────────────────────
            search_result = await two_level_search(
                query=enhanced_query,
                user_id=user_id,
                filters=filters,
                top_k=20,
                active_web_collection_ids=active_web_ids or None,
                selected_doc_ids=selected_doc_ids or None,
            )
            chunks = search_result["chunks"]
            matched_documents = search_result.get("matched_documents", [])

            logger.info(
                "[RAGAgent] Two-level search: %d matched docs, %d chunks",
                len(matched_documents),
                len(chunks),
            )

            # ── Rerank (skipped when config.rag_use_llm_reranking=False) ─
            reranked_chunks, rerank_tokens = await rerank(query=enhanced_query, chunks=chunks, top_k=8)
            tokens_used += rerank_tokens

            sources = self._extract_sources(reranked_chunks)
            conversation_context = task_config.conversation_history
            
            synthesis_prompt = self.prompts.synthesis_prompt(
                query=enhanced_query,  
                chunks=reranked_chunks,
                conversation_context=conversation_context,
                long_term_memory=task_config.long_term_memory,
            )
            
            synthesis_result = await self.llm.generate(
                prompt=synthesis_prompt,
                temperature=config.rag_temperature,
            )
            final_answer = synthesis_result.text
            tokens_used += synthesis_result.usage.get("total_tokens", 0)
            
            logger.info(f"[RAGAgent] Generated answer: {final_answer[:200]}...")

            api_calls = 2  # query_expansion + synthesis (always)
            api_calls += 1  # two_level_search (stage-1 + stage-2 internally)
            if config.rag_use_llm_reranking:
                api_calls += 1

            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=True,
                result=final_answer, 
                data={
                    "answer": final_answer,  
                    "chunks": reranked_chunks,
                    "sources": sources,
                    "query": original_query,
                    "matched_documents": matched_documents,
                },
                confidence_score=self._calculate_confidence(reranked_chunks),
                execution_metadata={"attempt_number": 1, "warnings": []},
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start_time) * 1000),
                    "tokens_used": tokens_used,
                    "api_calls_made": api_calls,
                },
                depends_on=list(task_config.dependency_outputs.keys()),
                metadata={
                    "sources": sources,
                    "model_used": config.rag_model,
                    "llm_reranking": config.rag_use_llm_reranking,
                },
            )

        except asyncio.TimeoutError:
            logger.warning(f"[RAGAgent] Timeout for input: {task_config}")
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=False,
                result="Search timed out. Partial results may be available.",
                data={"chunks": chunks, "partial": True},
                confidence_score=self._calculate_confidence(chunks),
                error="timeout",
                execution_metadata={"attempt_number": 1, "warnings": ["Timed out during retrieval"]},
                resource_usage={"time_taken_ms": int((time.perf_counter() - start_time) * 1000)},
                depends_on=list(task_config.dependency_outputs.keys()),
            )

        except Exception:
            logger.exception(f"[RAGAgent] Error for input: {task_config}")
            raise  





    async def _expand_query(self, query: str, entities: Dict[str, Any], dependency_outputs: Dict[str, Any]) -> tuple[str, int]:
        """
        Expand the original query using LLM to generate a more comprehensive search query.
        
        Args:
            query: Original user query
            entities: Extracted entities from the query
            dependency_outputs: Outputs from dependent agents that may provide additional context for query expansion
            
        Returns:
            Enhanced query string for better retrieval
        """
        
        try:
            prompt = self.prompts.query_expansion_prompt(query, entities, dependency_outputs)
            
            result = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,  
                max_tokens=150,   
            )
            expand_usage = result.usage.get("total_tokens", 0)
            response_text = result.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            enhanced_query = result.get("queries", query)
            
            if not enhanced_query or not enhanced_query.strip():
                logger.warning(f"[RAGAgent] Query expansion returned empty, using original query")
                return query, expand_usage
                
            return enhanced_query, expand_usage
            
        except json.JSONDecodeError as e:
            logger.warning(f"[RAGAgent] Failed to parse query expansion JSON: {e}. Using original query.")
            return query, 0
        except Exception as e:
            logger.warning(f"[RAGAgent] Query expansion failed: {e}. Using original query.")
            return query, 0



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
        seen: dict[str, int] = {}          
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            fname = meta.get("filename", "")
            page = meta.get("page")
            doc_id = chunk.get("doc_id")
            if not fname:
                continue
            if fname not in seen:
                seen[fname] = len(sources)
                sources.append(
                    {
                        "type": "document",
                        "source": fname,
                        "page": [page] if page else [],
                        "section": meta.get("section_title"),
                        "doc_id": doc_id,
                    }
                )
            else:
                if page and page not in sources[seen[fname]]["page"]:
                    sources[seen[fname]]["page"].append(page)
                if doc_id and not sources[seen[fname]].get("doc_id"):
                    sources[seen[fname]]["doc_id"] = doc_id
        for s in sources:
            pages = s["page"]
            if not pages:
                s["page"] = None
            elif len(pages) == 1:
                s["page"] = pages[0]
        return sources


    @staticmethod
    def _calculate_confidence(chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        scores = [c.get("score", 0.0) for c in chunks]
        return min(1.0, sum(scores) / max(len(scores), 1))

