"""
Integration-level tests for the Orchestrator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from utils.schemas import (
    AgentInput,
    AgentOutput,
    ResolvedAgentTask,
    ResolvedMasterOutput,
)
from core.orchestrator import Orchestrator
from core.state_manager import StateManager


def _make_plan(*agent_specs) -> ResolvedMasterOutput:
    """Helper: build a ResolvedMasterOutput from (agent_id, depends_on, task) tuples."""
    agents = []
    for agent_id, deps, task in agent_specs:
        agents.append(
            ResolvedAgentTask(
                agent_type=agent_id.split("_")[0] + "_agent",
                task=task,
                priority=1,
                depends_on_indices=[],
                agent_id=agent_id,
                depends_on=deps,
            )
        )
    return ResolvedMasterOutput(intent="test", agents=agents)


class TestOrchestratorExecution:
    @pytest.mark.asyncio
    async def test_single_agent(self):
        plan = _make_plan(("rag_agent_0", [], "search docs"))

        mock_memory = MagicMock()
        mock_memory.get_context = AsyncMock(return_value="")

        orch = Orchestrator(memory_manager=mock_memory)
        state = StateManager()

        with patch.object(orch, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = AgentOutput(
                agent_name="rag_agent",
                status="success",
                result="found docs",
            )
            results = await orch.execute(plan, state)

        assert "rag_agent_0" in results
        assert results["rag_agent_0"].status == "success"

    @pytest.mark.asyncio
    async def test_dependent_agents_run_in_order(self):
        plan = _make_plan(
            ("rag_agent_0", [], "search"),
            ("code_agent_1", ["rag_agent_0"], "process"),
        )

        mock_memory = MagicMock()
        mock_memory.get_context = AsyncMock(return_value="")

        orch = Orchestrator(memory_manager=mock_memory)
        state = StateManager()

        call_order = []

        async def mock_run(task_dict, context, state_mgr):
            call_order.append(task_dict["agent_id"])
            return AgentOutput(
                agent_name=task_dict["agent_type"],
                status="success",
                result=f"result for {task_dict['agent_id']}",
            )

        with patch.object(orch, "_run_agent", side_effect=mock_run):
            results = await orch.execute(plan, state)

        assert call_order == ["rag_agent_0", "code_agent_1"]
