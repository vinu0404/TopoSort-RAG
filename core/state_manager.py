"""
Shared state manager

Keeps agent outputs keyed by agent_id for cross-agent dependency resolution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.schemas import AgentOutput


class StateManager:
    def __init__(self):
        self._state: Dict[str, AgentOutput] = {}

    def set(self, agent_id: str, output: AgentOutput) -> None:
        self._state[agent_id] = output

    def get(self, agent_id: str) -> Optional[AgentOutput]:
        return self._state.get(agent_id)

    def get_data(self, agent_id: str) -> Any:
        """Get the data (or partial_data) for an upstream dependency."""
        output = self._state.get(agent_id)
        if output is None:
            return None
        if output.task_done:
            return output.data
        return output.partial_data

    def all(self) -> Dict[str, AgentOutput]:
        return dict(self._state)

    def clear(self) -> None:
        self._state.clear()
