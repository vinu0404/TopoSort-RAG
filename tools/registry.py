"""
Singleton ToolRegistry with auto-discovery.

BUG-9  FIX : uses pathlib for cross-platform path handling so it works on
             Windows (backslashes) and POSIX (forward slashes).
BUG-15 FIX : auto_discover_tools scans `tools/*_tools.py`.  Agent-level
             `agents/<name>/tools.py` files are *wrappers* that re-export
             functions from here; they are NOT scanned by discovery.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pathlib
from typing import Callable, Dict, List, Set

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Process-wide singleton that maps tool-name → callable + permissions."""

    _instance: "ToolRegistry | None" = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._tools: Dict[str, Callable] = {}
            inst._permissions: Dict[str, Set[str]] = {}
            cls._instance = inst
        return cls._instance


    def register(self, tool_name: str, tool_fn: Callable, allowed_agents: List[str]) -> None:
        self._tools[tool_name] = tool_fn
        self._permissions[tool_name] = set(allowed_agents)

    def get_tool(self, tool_name: str, agent_name: str) -> Callable:
        """
        Return the tool callable if `agent_name` is authorised.

        Raises
        ------
        ValueError      – tool not found
        PermissionError – agent not in allow-list
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        if agent_name not in self._permissions.get(tool_name, set()):
            raise PermissionError(
                f"Agent '{agent_name}' is not authorised to use tool '{tool_name}'"
            )
        return self._tools[tool_name]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    # ── auto-discovery ──────────────────────────────────────────────────

    def auto_discover_tools(self, tools_dir: str = "tools") -> None:
        """
        Scan ``tools/*_tools.py`` for functions decorated with ``@tool``.
        """
        tools_path = pathlib.Path(tools_dir).resolve()

        tool_files = sorted(tools_path.glob("*_tools.py"))

        if not tool_files:
            raise RuntimeError(f"No *_tools.py files found in {tools_path}")

        for tool_file in tool_files:
            try:
                relative = tool_file.relative_to(tools_path.parent)
                module_name = ".".join(relative.with_suffix("").parts)

                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if getattr(obj, "is_tool", False):
                        if not hasattr(obj, "allowed_agents"):
                            raise RuntimeError(
                                f"Tool '{name}' in {tool_file.name} is missing "
                                f"'allowed_agents' (bad @tool usage)"
                            )
                        self.register(
                            tool_name=name,
                            tool_fn=obj,
                            allowed_agents=obj.allowed_agents,
                        )

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load tools from {tool_file.name}: {exc}"
                ) from exc

        logger.info(
            "Registered %d tools from %d files",
            len(self._tools),
            len(tool_files),
        )

    # ── reset (for tests) ──────────────────────────────────────────────

    @classmethod
    def reset(cls) -> None:
        """Destroy singleton — only useful in test teardown."""
        cls._instance = None
