"""
Master Agent — single LLM call that analyses the user query,
extracts entities, classifies intent, and builds the execution plan.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

from config.model_list import AgentRegistry
from config.settings import config
from utils.llm_providers import BaseLLMProvider, get_llm_provider
from utils.schemas import (
    AgentTask,
    Analysis,
    ExecutionPlan,
    MasterAgentOutput,
    ResolvedAgentTask,
    ResolvedExecutionPlan,
    ResolvedMasterOutput,
)

logger = logging.getLogger(__name__)


class MasterAgent:
    def __init__(
        self,
        agent_registry: AgentRegistry,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.agent_registry = agent_registry
        self.llm = llm_provider or get_llm_provider(
            config.master_model_provider,
            default_model=config.master_model,
        )

    async def plan(
        self,
        query: str,
        user_id: str,
        *,
        conversation_history: list | None = None,
        long_term_memory: dict | None = None,
    ) -> ResolvedMasterOutput:
        """
        Single call → analyse + plan.

        Returns a `ResolvedMasterOutput` with proper agent_ids assigned.
        """
        query_id = str(uuid.uuid4())
        agent_capabilities = self.agent_registry.get_agent_capabilities()

        prompt = self._build_prompt(
            query=query,
            agent_capabilities=agent_capabilities,
            conversation_history=conversation_history or [],
            long_term_memory=long_term_memory or {},
        )

        logger.info(f"[MasterAgent] Planning for query_id={query_id}, user_id={user_id}, query={query}")
        logger.debug(f"[MasterAgent] Prompt: {prompt[:500]}")
        raw: Dict[str, Any] = await self.llm.generate(
            prompt=prompt,
            temperature=config.master_temperature,
            model=config.master_model,
            output_schema={
                "analysis": {
                    "intent": "string",
                    "complexity": "string",
                    "entities": {
                        "date_range": "string or null",
                        "metric": "string or null",
                        "names": ["string"],
                        "locations": ["string"],
                        "doc_type": "string or null",
                    },
                    "constraints": {
                        "time_bound": "boolean",
                        "format_required": "string or null",
                        "confidence_threshold": "float",
                    },
                },
                "execution_plan": {
                    "agents": [
                        {
                            "agent_name": "string",
                            "task": "string",
                            "entities": {},
                            "tools": ["string"],
                            "depends_on_indices": ["int"],
                            "timeout": "int",
                            "max_retries": "int",
                            "priority": "string",
                        }
                    ],
                    "estimated_time": "int",
                    "estimated_cost": "float",
                },
            },
        )
        if not isinstance(raw, dict):
            raise ValueError(f"MasterAgent did not return a dict: {type(raw)}")

            logger.info(f"[MasterAgent] Output for query_id={query_id}: {raw}")
        master_output = MasterAgentOutput(
            query_id=query_id,
            user_id=user_id,
            original_query=query,
            analysis=Analysis(**raw.get("analysis", {})),
            execution_plan=ExecutionPlan(
                agents=[AgentTask(**a) for a in raw.get("execution_plan", {}).get("agents", [])],
                estimated_time=raw.get("execution_plan", {}).get("estimated_time", 0),
                estimated_cost=raw.get("execution_plan", {}).get("estimated_cost", 0.0),
            ),
        )
        return generate_agent_ids(master_output)

    @staticmethod
    def _build_prompt(
        query: str,
        agent_capabilities: List[Dict[str, Any]],
        conversation_history: list,
        long_term_memory: dict,
    ) -> str:
        agent_ref = ""
        for agent in agent_capabilities:
            name = agent.get("agent_name") or agent.get("name", "unknown")
            desc = agent.get("description", "")
            tools = ", ".join(agent.get("tools", []))
            uses = "\n".join(f"    - {u}" for u in agent.get("use_cases") or agent.get("typical_use_cases", []))
            agent_ref += f"\n**{name}**: {desc}\n  Tools: [{tools}]\n  Use cases:\n{uses}\n"

        conv_section = ""
        if conversation_history:
            conv_section = "### Recent Conversation\n"
            for turn in conversation_history[-6:]:
                role = turn.get("role", "user")
                content = str(turn.get("content", ""))[:300]
                conv_section += f"  {role}: {content}\n"

        memory_section = ""
        if long_term_memory:
            memory_section = f"### User Profile\n{long_term_memory}\n"

        return f"""You are the Master Planning Agent for a multi-agent RAG system.
Your job is to analyse the user's query, extract entities, and create an optimal execution plan.

### User Query
{query}

### Available Agents
{agent_ref}

{conv_section}
{memory_section}

### Planning Instructions

**Step 1 — Entity Extraction** (LLM-based, no regex):
- Dates / date ranges (e.g. "last quarter" → "2025-Q4", "January 2026" → "2026-01")
- People names, organisation names
- Locations
- Metrics / KPIs mentioned
- Document types (PDF, report, spreadsheet, etc.)
- Email addresses, URLs

**Step 2 — Intent Classification**:
Choose the primary intent:
  `data_retrieval` | `code_generation` | `email_search` | `email_composition` |
  `web_research` | `multi_step_analysis` | `comparison` | `summarisation`

**Step 3 — Complexity Assessment**:
  `simple` — single agent, no dependencies
  `medium` — 2 agents, possibly with one dependency
  `complex` — 3+ agents, multi-step dependencies

**Step 4 — Execution Plan**:
- Select the minimum set of agents needed. Don't over-plan.
- For Gmail tasks: use **mail_agent** (it can search inbox, sent, drafts, send, draft, reply).
- For web lookups: use **web_search_agent** (Tavily-powered search + URL extraction).
- For document questions: use **rag_agent**.
- For calculations / code tasks: use **code_agent**.
- Specify `depends_on_indices` as integer indices referencing earlier agents in the array.
  Agent at index 0 has no dependencies. Agent at index 1 can depend on [0], etc.
- Only assign tools that the agent actually has access to.
- If the same agent type is needed for multiple independent tasks, list it multiple times.

**Step 5 — Priority (per agent)**:
  `critical` — must succeed for the answer | `high` — important but not blocking |
  `medium` — nice to have | `low` — optional enrichment

### Output (JSON)
Return EXACTLY this structure:
{{
    "analysis": {{
        "intent": "<intent>",
        "complexity": "simple | medium | complex",
        "entities": {{
            "date_range": "<extracted or null>",
            "metric": "<extracted or null>",
            "names": ["<name1>", ...],
            "locations": ["<loc1>", ...],
            "doc_type": "<type or null>"
        }},
        "constraints": {{
            "time_bound": true | false,
            "format_required": "<format or null>",
            "confidence_threshold": 0.7
        }}
    }},
    "execution_plan": {{
        "agents": [
            {{
                "agent_name": "<agent_name>",
                "task": "<specific task description>",
                "entities": {{}},
                "tools": ["<tool1>", ...],
                "depends_on_indices": [],
                "timeout": 30,
                "max_retries": 2,
                "priority": "critical | high | medium | low"
            }}
        ],
        "estimated_time": <seconds>,
        "estimated_cost": <usd_estimate>
    }}
}}"""




def generate_agent_ids(master_output: MasterAgentOutput) -> ResolvedMasterOutput:
    """
    Convert index-based deps (`depends_on_indices`) into concrete
    agent-id-based deps (`depends_on`), producing a wholly new
    `ResolvedMasterOutput` with `ResolvedAgentTask` items.

    *No* runtime mutation of frozen Pydantic models.
    *No* delattr / setattr.
    """
    index_to_id: Dict[int, str] = {}
    resolved_tasks: List[ResolvedAgentTask] = []

    for idx, task in enumerate(master_output.execution_plan.agents):
        agent_id = f"{task.agent_name}_{uuid.uuid4().hex[:8]}"
        index_to_id[idx] = agent_id

    for idx, task in enumerate(master_output.execution_plan.agents):
        resolved_tasks.append(
            ResolvedAgentTask(
                agent_id=index_to_id[idx],
                agent_name=task.agent_name,
                task=task.task,
                entities=task.entities,
                tools=task.tools,
                depends_on=[
                    index_to_id[dep_idx]
                    for dep_idx in task.depends_on_indices
                    if dep_idx in index_to_id
                ],
                timeout=task.timeout,
                max_retries=task.max_retries,
                priority=task.priority,
            )
        )

    return ResolvedMasterOutput(
        query_id=master_output.query_id,
        user_id=master_output.user_id,
        original_query=master_output.original_query,
        analysis=master_output.analysis,
        execution_plan=ResolvedExecutionPlan(
            agents=resolved_tasks,
            estimated_time=master_output.execution_plan.estimated_time,
            estimated_cost=master_output.execution_plan.estimated_cost,
        ),
        metadata=master_output.metadata,
    )
