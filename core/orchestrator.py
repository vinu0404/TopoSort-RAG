"""
Orchestrator — executes the resolved agent plan stage by stage.

Supports optional HITL (Human-in-the-Loop) approval:  when an agent's
assigned tools include any marked ``requires_approval=True``, the
orchestrator pauses execution and delegates to an ``on_hitl_needed``
callback (provided by the streaming layer) which handles DB persistence,
SSE notification, and polling for the user's decision.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from tools.registry import ToolRegistry
from utils.dependency_resolver import resolve_dependencies
from utils.schemas import AgentInput, AgentOutput, HitlResolvedDecision, ResolvedExecutionPlan

logger = logging.getLogger(__name__)

# Type alias for the HITL callback supplied by the streaming / route layer.
# Signature:  (agent_cfg, hitl_tool_names) -> HitlResolvedDecision
HitlCallback = Callable[
    [Dict[str, Any], List[str]],
    Coroutine[Any, Any, HitlResolvedDecision],
]


class Orchestrator:
    def __init__(
        self,
        agent_instances: Dict[str, Any],
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """
        Parameters
        ----------
        agent_instances : mapping of agent_name → agent object
                          e.g. {"rag_agent": <RAGAgent>, ...}
        tool_registry   : optional ToolRegistry for HITL look-ups.
                          Falls back to the singleton if not provided.
        """
        self.agent_instances = agent_instances
        self.shared_state: Dict[str, AgentOutput | Dict[str, Any]] = {}
        self._tool_registry = tool_registry or ToolRegistry()

    # ── public entry point ──────────────────────────────────────────────

    async def execute_plan(
        self,
        execution_plan: ResolvedExecutionPlan,
        context: Dict[str, Any],
        on_hitl_needed: Optional[HitlCallback] = None,
    ) -> Dict[str, Any]:
        """
        Execute the plan stage-by-stage.

        Parameters
        ----------
        on_hitl_needed : async callback that is invoked when an agent has
            tools requiring approval.  If *None*, HITL agents are
            auto-skipped with ``error="hitl_skipped_non_streaming"``.
        """
        agent_dicts = [task.model_dump() for task in execution_plan.agents]
        stages = resolve_dependencies(agent_dicts)

        logger.info("[Orchestrator] Starting execution plan with %d agents.", len(agent_dicts))
        for stage_num, stage_agents in enumerate(stages, start=1):
            await self._execute_stage(stage_num, stage_agents, context, on_hitl_needed)

        return self.shared_state

    # ── stage execution ─────────────────────────────────────────────────

    async def _execute_stage(
        self,
        stage_num: int,
        stage_agents: List[Dict[str, Any]],
        context: Dict[str, Any],
        on_hitl_needed: Optional[HitlCallback],
    ) -> None:
        logger.info("Stage %d — running %d agent(s)", stage_num, len(stage_agents))

        coroutines = []
        agent_ids: List[str] = []

        for agent_cfg in stage_agents:
            agent_name = agent_cfg["agent_name"]
            agent_id = agent_cfg["agent_id"]
            agent_ids.append(agent_id)

            agent = self.agent_instances.get(agent_name)
            if agent is None:
                logger.error("No instance for agent '%s' — skipping", agent_name)
                self.shared_state[agent_id] = AgentOutput(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    task_description=agent_cfg.get("task", ""),
                    task_done=False,
                    error=f"No agent instance registered for '{agent_name}'",
                )
                continue

            coroutines.append(
                (agent_id, self._run_agent_with_hitl(agent_cfg, agent, context, on_hitl_needed))
            )

        if coroutines:
            results = await asyncio.gather(
                *[coro for _, coro in coroutines],
                return_exceptions=True,
            )
            for (aid, _), result in zip(coroutines, results):
                if isinstance(result, Exception):
                    logger.error("Agent %s raised: %s", aid, result)
                    self.shared_state[aid] = AgentOutput(
                        agent_id=aid,
                        agent_name="unknown",
                        task_description="",
                        task_done=False,
                        error=str(result),
                    )
                else:
                    logger.info("[Orchestrator] Output from agent %s: %s", aid, result)
                    self.shared_state[aid] = result

    # ── HITL-aware agent runner ─────────────────────────────────────────

    async def _run_agent_with_hitl(
        self,
        agent_cfg: Dict[str, Any],
        agent: Any,
        context: Dict[str, Any],
        on_hitl_needed: Optional[HitlCallback],
    ) -> AgentOutput:
        """
        Check if any of the agent's assigned tools require HITL approval.
        If so, invoke the callback and act on the decision.
        Otherwise run the agent directly.
        """
        agent_id = agent_cfg["agent_id"]
        agent_name = agent_cfg["agent_name"]
        task = agent_cfg.get("task", "")

        # Determine which tools (if any) need approval
        hitl_tools = self._tool_registry.get_hitl_tools_for_agent_task(
            agent_cfg.get("tools", [])
        )

        hitl_decision: Optional[HitlResolvedDecision] = None

        if hitl_tools:
            if on_hitl_needed is None:
                # Non-streaming endpoint — auto-skip HITL agents
                logger.warning(
                    "Agent %s (%s) requires HITL for tools %s but no callback — skipping",
                    agent_id, agent_name, hitl_tools,
                )
                return AgentOutput(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    task_description=task,
                    task_done=False,
                    error="hitl_skipped_non_streaming",
                    partial_data={
                        "reason": "This agent requires human approval. Use the streaming endpoint.",
                        "hitl_tools": hitl_tools,
                    },
                )

            # Invoke the HITL callback — blocks until user responds or timeout
            hitl_decision = await on_hitl_needed(agent_cfg, hitl_tools)

            if not hitl_decision.approved:
                logger.info(
                    "HITL denied for agent %s (%s): %s",
                    agent_id, agent_name, hitl_decision.reason,
                )
                return AgentOutput(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    task_description=task,
                    task_done=False,
                    error=f"denied_by_user" if hitl_decision.reason == "denied_by_user" else hitl_decision.reason or "hitl_denied",
                    partial_data={
                        "reason": hitl_decision.reason or "denied_by_user",
                        "hitl_tools": hitl_tools,
                    },
                )

            logger.info(
                "HITL approved for agent %s (%s), instructions=%s",
                agent_id, agent_name, hitl_decision.instructions,
            )

        # Build input (with HITL context if available)
        agent_input = self._prepare_agent_input(agent_cfg, context, hitl_decision)
        return await agent.execute_with_retry(agent_input)

    # ── input builder ───────────────────────────────────────────────────

    def _prepare_agent_input(
        self,
        agent_cfg: Dict[str, Any],
        context: Dict[str, Any],
        hitl_decision: Optional[HitlResolvedDecision] = None,
    ) -> AgentInput:
        """Build a typed AgentInput from the resolved agent dict + context."""
        dependency_outputs: Dict[str, Any] = {}
        for dep_id in agent_cfg.get("depends_on", []):
            dep_output = self.shared_state.get(dep_id)
            if dep_output is None:
                dependency_outputs[dep_id] = None
            elif isinstance(dep_output, AgentOutput):
                if dep_output.task_done:
                    dependency_outputs[dep_id] = dep_output.data
                elif dep_output.partial_data is not None:
                    dependency_outputs[dep_id] = dep_output.partial_data
                else:
                    dependency_outputs[dep_id] = None
            elif isinstance(dep_output, dict):
                if dep_output.get("task_done"):
                    dependency_outputs[dep_id] = dep_output.get("data")
                elif dep_output.get("partial_data"):
                    dependency_outputs[dep_id] = dep_output.get("partial_data")
                else:
                    dependency_outputs[dep_id] = None

        return AgentInput(
            agent_id=agent_cfg["agent_id"],
            agent_name=agent_cfg["agent_name"],
            task=agent_cfg.get("task", ""),
            entities=agent_cfg.get("entities", {}),
            tools=agent_cfg.get("tools", []),
            conversation_history=context.get("conversation_history", []),
            long_term_memory=context.get("long_term_memory", {}),
            dependency_outputs=dependency_outputs,
            retry_config={
                "max_retries": agent_cfg.get("max_retries", 2),
                "timeout": agent_cfg.get("timeout", 30),
                "backoff_multiplier": 2.0,
            },
            metadata={
                "user_id": context.get("user_id", ""),
                "query_id": context.get("query_id", ""),
                "session_id": context.get("session_id"),
            },
            hitl_context=hitl_decision,
        )
