"""
Tests for the dependency resolver (Kahn's topological sort).
"""

import pytest

from utils.dependency_resolver import resolve_dependencies


def _agent(agent_id: str, depends_on: list[str] | None = None) -> dict:
    return {"agent_id": agent_id, "depends_on": depends_on or []}


class TestResolveDependencies:
    def test_no_dependencies(self):
        agents = [_agent("a"), _agent("b"), _agent("c")]
        stages = resolve_dependencies(agents)
        assert len(stages) == 1
        ids = {a["agent_id"] for a in stages[0]}
        assert ids == {"a", "b", "c"}

    def test_linear_chain(self):
        agents = [_agent("a"), _agent("b", ["a"]), _agent("c", ["b"])]
        stages = resolve_dependencies(agents)
        assert len(stages) == 3
        assert stages[0][0]["agent_id"] == "a"
        assert stages[1][0]["agent_id"] == "b"
        assert stages[2][0]["agent_id"] == "c"

    def test_diamond_dependency(self):
        agents = [
            _agent("a"),
            _agent("b", ["a"]),
            _agent("c", ["a"]),
            _agent("d", ["b", "c"]),
        ]
        stages = resolve_dependencies(agents)
        assert len(stages) == 3
        stage_0_ids = {a["agent_id"] for a in stages[0]}
        stage_1_ids = {a["agent_id"] for a in stages[1]}
        stage_2_ids = {a["agent_id"] for a in stages[2]}
        assert stage_0_ids == {"a"}
        assert stage_1_ids == {"b", "c"}
        assert stage_2_ids == {"d"}

    def test_circular_dependency_raises(self):
        agents = [_agent("a", ["b"]), _agent("b", ["a"])]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_dependencies(agents)
