"""
Runtime validators used at startup and during query execution.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from utils.schemas import ResolvedAgentTask

logger = logging.getLogger(__name__)


def validate_tools_for_agents(
    agent_instances: Dict,
    tool_registry,
) -> None:
    """
    Called once at startup.  For every registered agent, verify that the
    ToolRegistry contains every tool the agent declares as *required*.
    """
    for agent_name, agent in agent_instances.items():
        for tool_name in agent.get_required_tools():
            try:
                tool_registry.get_tool(tool_name, agent_name)
            except (ValueError, PermissionError) as exc:
                raise RuntimeError(
                    f"Startup validation failed — agent '{agent_name}' requires "
                    f"tool '{tool_name}' but: {exc}"
                ) from exc

    logger.info("All agent→tool requirements validated")


def validate_execution_plan(agents: List[ResolvedAgentTask]) -> None:
    """Quick sanity checks on a resolved execution plan."""
    ids = {a.agent_id for a in agents}
    for agent in agents:
        for dep in agent.depends_on:
            if dep not in ids:
                raise ValueError(
                    f"Agent '{agent.agent_id}' depends on '{dep}' which is "
                    f"not present in the execution plan."
                )
