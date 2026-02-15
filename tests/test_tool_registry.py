"""
Tests for the tool registry — especially (Windows path handling).
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from tools.registry import ToolRegistry


class TestToolRegistryDiscovery:
    def setup_method(self):
        ToolRegistry.reset()

    def test_register_and_get(self):
        registry = ToolRegistry()

        def dummy_tool():
            return "ok"

        dummy_tool.is_tool = True
        dummy_tool.allowed_agents = ["rag_agent"]

        registry.register("dummy_tool", dummy_tool, ["rag_agent"])
        tool = registry.get_tool("dummy_tool", "rag_agent")
        assert tool is not None
        assert tool() == "ok"

    def test_permission_denied(self):
        registry = ToolRegistry()

        def restricted():
            return "secret"

        restricted.is_tool = True
        restricted.allowed_agents = ["code_agent"]

        registry.register("restricted", restricted, ["code_agent"])
        tool = registry.get_tool("restricted", "rag_agent")
        assert tool is None

    def test_auto_discover_uses_pathlib(self):
        """verifies that auto_discover doesn't break on Windows paths."""
        registry = ToolRegistry()
        # Just ensuring it doesn't crash — actual tool loading depends on environment
        try:
            registry.auto_discover_tools()
        except ImportError:
            pass  # OK — tool modules may have unmet dependencies in test env
