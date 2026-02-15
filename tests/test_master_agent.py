"""
"""

import pytest
from unittest.mock import AsyncMock, patch

from utils.schemas import (
    AgentTask,
    ExecutionPlan,
    MasterAgentOutput,
    ResolvedAgentTask,
    ResolvedExecutionPlan,
    ResolvedMasterOutput,
)
from core.master_agent import generate_agent_ids


class TestGenerateAgentIds:
    """Verify the two-pass BUG-1 fix works correctly."""

    def test_basic_id_generation(self):
        raw = MasterAgentOutput(
            intent="test query",
            plan=ExecutionPlan(
                agents=[
                    AgentTask(
                        agent_type="rag_agent",
                        task="Search documents",
                        priority=1,
                        depends_on_indices=[],
                    ),
                    AgentTask(
                        agent_type="code_agent",
                        task="Execute code",
                        priority=2,
                        depends_on_indices=[0],
                    ),
                ],
            ),
        )

        resolved = generate_agent_ids(raw)

        assert isinstance(resolved, ResolvedMasterOutput)
        assert len(resolved.agents) == 2

        rag = resolved.agents[0]
        code = resolved.agents[1]
        assert rag.agent_id.startswith("rag_agent_")
        assert code.agent_id.startswith("code_agent_")
        assert code.depends_on == [rag.agent_id]
        assert rag.depends_on == []

    def test_no_mutation_of_original(self):
        raw = MasterAgentOutput(
            intent="test",
            plan=ExecutionPlan(
                agents=[
                    AgentTask(agent_type="rag_agent", task="t", priority=1, depends_on_indices=[]),
                ],
            ),
        )

        resolved = generate_agent_ids(raw)

        # Original unchanged
        assert not hasattr(raw.plan.agents[0], "agent_id")
        # Resolved has the field
        assert resolved.agents[0].agent_id is not None
