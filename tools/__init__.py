"""
@tool decorator — marks a function as a tool and declares which agents
may invoke it.

Usage:
    from tools import tool

    @tool("rag_agent", "web_search_agent")
    async def web_search(query: str) -> List[Dict]:
        ...

    @tool("mail_agent", requires_approval=True)
    async def send_email(...) -> Dict:
        ...
"""

from __future__ import annotations

from typing import Callable, List


def tool(
    *allowed_agents: str,
    requires_approval: bool = False,
) -> Callable:
    """
    Decorator that tags a function as a registered tool.

    Parameters
    ----------
    *allowed_agents : agent names that are permitted to call this tool.
    requires_approval : if True, the orchestrator will request HITL
        (Human-in-the-Loop) approval before the owning agent executes.
    """

    def decorator(func: Callable) -> Callable:
        func.is_tool = True  # type: ignore[attr-defined]
        func.allowed_agents: List[str] = list(allowed_agents)  # type: ignore[attr-defined]
        func.requires_approval: bool = requires_approval  # type: ignore[attr-defined]
        return func

    return decorator
