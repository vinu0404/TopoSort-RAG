"""
Topological sort using Kahn's algorithm.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List


def resolve_dependencies(agents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Parameters
    ----------
    agents : list of dicts, each with at least::
        {"agent_id": str, "depends_on": List[str]}

    Returns
    -------
    List of stages.  Each stage is a list of agent dicts that can run
    in parallel.  Stages are ordered so every dependency of a stage-N
    agent has completed in a stage < N.
    """
    if not agents:
        return []

    graph: Dict[str, List[str]] = defaultdict(list) 
    in_degree: Dict[str, int] = {}
    agent_map: Dict[str, Dict[str, Any]] = {}

    for agent in agents:
        aid = agent["agent_id"]
        agent_map[aid] = agent
        in_degree[aid] = len(agent.get("depends_on", []))
        for dep in agent.get("depends_on", []):
            graph[dep].append(aid)

    queue: deque[str] = deque(
        aid for aid, deg in in_degree.items() if deg == 0
    )

    stages: List[List[Dict[str, Any]]] = []

    while queue:
        current_stage: List[Dict[str, Any]] = []
        for _ in range(len(queue)):
            aid = queue.popleft()
            current_stage.append(agent_map[aid])
            for dependent in graph[aid]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        stages.append(current_stage)

    remaining = {k: v for k, v in in_degree.items() if v > 0}
    if remaining:
        raise ValueError(f"Circular dependency detected among: {remaining}")

    return stages
