"""
AgentRegistry — loads config/agent_registry.yaml and exposes agent
capabilities to MasterAgent and the startup validator.
"""

import pathlib
from typing import Any, Dict, List

import yaml


class AgentRegistry:
    def __init__(self, registry_path: str | None = None):
        if registry_path is None:
            registry_path = str(
                pathlib.Path(__file__).parent / "agent_registry.yaml"
            )
        with open(registry_path, "r", encoding="utf-8") as fh:
            self.registry: Dict[str, Any] = yaml.safe_load(fh)

    def get_agent_capabilities(self) -> List[Dict[str, Any]]:
        """Format suitable for injection into the Master Agent prompt."""
        capabilities: list[dict] = []
        for agent_name, info in self.registry["agents"].items():
            capabilities.append(
                {
                    "agent_name": agent_name,
                    "description": info["description"],
                    "capabilities": info["capabilities"],
                    "tools": info["tools"],
                    "use_cases": info["typical_use_cases"],
                }
            )
        return capabilities

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        if agent_name not in self.registry["agents"]:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.registry["agents"][agent_name]

    def list_agent_names(self) -> List[str]:
        return list(self.registry["agents"].keys())
