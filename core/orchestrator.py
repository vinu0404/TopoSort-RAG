"""
Orchestrator — executes the resolved agent plan stage by stage.

"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from utils.dependency_resolver import resolve_dependencies
from utils.schemas import AgentInput, AgentOutput, ResolvedExecutionPlan

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, agent_instances: Dict[str, Any]):
        """
        Parameters
        ----------
        agent_instances : mapping of agent_name → agent object
                          e.g. {"rag_agent": <RAGAgent>, ...}
        """
        self.agent_instances = agent_instances
        self.shared_state: Dict[str, AgentOutput | Dict[str, Any]] = {}

    async def execute_plan(
        self,
        execution_plan: ResolvedExecutionPlan,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run all stages sequentially; agents within each stage in parallel.

        Returns the shared_state mapping agent_id → AgentOutput.
        """
        agent_dicts = [task.model_dump() for task in execution_plan.agents]
        stages = resolve_dependencies(agent_dicts)

        logger.info(f"[Orchestrator] Starting execution plan with {len(agent_dicts)} agents.")
        logger.debug(f"[Orchestrator] Context: {context}")
        for stage_num, stage_agents in enumerate(stages, start=1):
            await self._execute_stage(stage_num, stage_agents, context)

        return self.shared_state

    async def _execute_stage(
        self,
        stage_num: int,
        stage_agents: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> None:
        logger.info("Stage %d — running %d agent(s)", stage_num, len(stage_agents))

        coroutines = []
        agent_ids: List[str] = []

        for agent_cfg in stage_agents:
            agent_name = agent_cfg["agent_name"]
            agent_id = agent_cfg["agent_id"]
            agent_ids.append(agent_id)

            logger.info(f"[Orchestrator] Input to agent {agent_name} ({agent_id}): {agent_cfg}")
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

            agent_input = self._prepare_agent_input(agent_cfg, context)
            coroutines.append((agent_id, agent.execute_with_retry(agent_input)))

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
                    logger.info(f"[Orchestrator] Output from agent {aid}: {result}")
                    self.shared_state[aid] = result

    def _prepare_agent_input(
        self, agent_cfg: Dict[str, Any], context: Dict[str, Any]
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
        )
