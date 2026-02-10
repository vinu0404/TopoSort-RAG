"""
Pydantic schemas for the entire Multi-Agentic RAG system.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Master Agent — Input / Output
# ═══════════════════════════════════════════════════════════════════════════════


class EntityAnalysis(BaseModel):
    date_range: Optional[str] = None
    metric: Optional[str] = None
    names: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    doc_type: Optional[str] = None


class Constraints(BaseModel):
    time_bound: bool = False
    format_required: Optional[str] = None
    confidence_threshold: float = 0.7


class Analysis(BaseModel):
    intent: str
    complexity: str  # "simple" | "medium" | "complex"
    entities: EntityAnalysis
    constraints: Constraints


class AgentTask(BaseModel):
    """
    Raw plan item returned by the Master LLM.

    Uses *indices* for dependency references because the LLM doesn't know
    agent_ids yet.  See ResolvedAgentTask for the post-processed version.
    """

    agent_name: str
    task: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)
    depends_on_indices: List[int] = Field(default_factory=list)
    timeout: int = 30
    max_retries: int = 2
    priority: str = "medium"  # "high" | "medium" | "low"


class ResolvedAgentTask(BaseModel):
    """
    BUG-1 FIX — Post-processed version of AgentTask.

    Created by `generate_agent_ids()` which converts index-based deps
    into real agent_id strings.  This is what Orchestrator / tests use.
    """

    agent_id: str
    agent_name: str
    task: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    timeout: int = 30
    max_retries: int = 2
    priority: str = "medium"


class ExecutionPlan(BaseModel):
    agents: List[AgentTask] = Field(default_factory=list)
    estimated_time: int = 0
    estimated_cost: float = 0.0


class ResolvedExecutionPlan(BaseModel):
    """Plan with agent_ids assigned and index-deps resolved."""

    agents: List[ResolvedAgentTask] = Field(default_factory=list)
    estimated_time: int = 0
    estimated_cost: float = 0.0


class MasterAgentOutput(BaseModel):
    query_id: str
    user_id: str
    original_query: str
    analysis: Analysis
    execution_plan: ExecutionPlan
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResolvedMasterOutput(BaseModel):
    """After generate_agent_ids() transforms the plan."""

    query_id: str
    user_id: str
    original_query: str
    analysis: Analysis
    execution_plan: ResolvedExecutionPlan
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Universal Agent Input / Output
# ═══════════════════════════════════════════════════════════════════════════════


class AgentInput(BaseModel):
    """Universal input schema for ALL agents."""

    agent_id: str = Field(..., description="Unique agent instance ID")
    agent_name: str = Field(..., description="Agent type (rag_agent, code_agent, …)")
    task: str = Field(..., description="Specific task description for this agent")

    entities: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)

    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    long_term_memory: Dict[str, Any] = Field(default_factory=dict)

    dependency_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key = upstream agent_id, Value = its data or partial_data",
    )

    retry_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_retries": 2,
            "timeout": 30,
            "backoff_multiplier": 2.0,
        }
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """Universal output schema for ALL agents."""

    agent_id: str
    agent_name: str
    task_description: str
    task_done: bool

    data: Optional[Dict[str, Any]] = Field(None, description="Full results if success")
    partial_data: Optional[Dict[str, Any]] = Field(
        None, description="Partial results on timeout / failure"
    )

    error: Optional[str] = None
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)

    execution_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "attempt_number": 1,
            "timeout_used": 30,
            "execution_order": 0,
            "parallel_group": None,
            "parent_task_id": None,
            "warnings": [],
        }
    )

    resource_usage: Dict[str, Any] = Field(
        default_factory=lambda: {
            "time_taken_ms": 0,
            "tokens_used": 0,
            "api_calls_made": 0,
            "cache_hits": 0,
            "estimated_cost": 0.0,
        }
    )

    depends_on: List[str] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "sources": [],
            "citations": [],
            "model_used": None,
            "timestamp": None,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class ConversationTurn(BaseModel):
    turn: int
    user_query: str
    composer_answer: str
    timestamp: str


class ConversationSummary(BaseModel):
    """Generated every N turns (default 3)."""

    turns: List[int]  # e.g. [1, 2, 3]
    summary: str
    key_points: List[str] = Field(default_factory=list)
    timestamp: str


class AgentConversationHistory(BaseModel):
    """
    What agents receive as conversation context.

    BUG-14 FIX: `recent_unsummarized_turns` carries turns that have NOT yet
    been grouped into a summary (e.g. turns 4 & 5 before turn 6 completes
    the next 3-turn group).  Agents therefore *always* see the latest context.
    """

    summaries: List[ConversationSummary] = Field(default_factory=list)
    recent_turns: List[ConversationTurn] = Field(default_factory=list)
    recent_unsummarized_turns: List[ConversationTurn] = Field(default_factory=list)


class CriticalFacts(BaseModel):
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    recent_projects: List[str] = Field(default_factory=list)
    user_name: Optional[str] = None


class UserPreferences(BaseModel):
    tone: str = "professional"
    detail_level: str = "concise"
    language: str = "en"


class LongTermMemory(BaseModel):
    user_id: str
    critical: CriticalFacts = Field(default_factory=CriticalFacts)
    preferences: UserPreferences = Field(default_factory=UserPreferences)


# ═══════════════════════════════════════════════════════════════════════════════
# Composer Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class Source(BaseModel):
    type: str  # "document" | "web" | "database"
    agent: str  # agent_id
    source: str  # filename or URL
    page: Optional[int] = None
    url: Optional[str] = None
    excerpt: Optional[str] = None


class ComposerInput(BaseModel):
    query_id: str
    original_query: str
    user_id: str

    execution_summary: Dict[str, Any] = Field(default_factory=dict)
    agent_results: List[AgentOutput] = Field(default_factory=list)
    all_sources: List[Source] = Field(default_factory=list)

    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    long_term_memory: LongTermMemory

    timestamp: str = Field(default_factory=lambda: str(time.time()))


class ComposerOutput(BaseModel):
    query_id: str
    answer: str
    sources: List[Source] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# API Request / Response helpers
# ═══════════════════════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    query: str
    user_id: str = Field(..., description="User ID for memory and document access")
    session_id: Optional[str] = None


class MasterAgentInput(BaseModel):
    query_id: str
    user_id: str
    user_query: str
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    long_term_memory: LongTermMemory
    available_agents: List[Dict[str, Any]] = Field(default_factory=list)
