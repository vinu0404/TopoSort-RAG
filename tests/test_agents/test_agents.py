"""Tests for individual agents — RAG, Code, Mail, and WebSearch."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from utils.schemas import AgentInput, AgentOutput


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_input(**overrides) -> AgentInput:
    """Build a minimal AgentInput with sensible defaults."""
    defaults = dict(
        task="test task",
        context="",
        tools_available=[],
        dependencies={},
    )
    defaults.update(overrides)
    return AgentInput(**defaults)


def _mock_tool_registry(*tool_names: str) -> MagicMock:
    """Return a ToolRegistry mock whose get_tool returns AsyncMock callables."""
    registry = MagicMock()
    tool_map = {name: AsyncMock(return_value=[]) for name in tool_names}
    registry.get_tool = MagicMock(side_effect=lambda name, _agent: tool_map.get(name, AsyncMock()))
    return registry


# ── RAG Agent ──────────────────────────────────────────────────────────────────


class TestRAGAgent:
    @pytest.mark.asyncio
    async def test_execute_returns_agent_output(self):
        from agents.rag_agent.agent import RAGAgent

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value='{"answer": "test answer", "sources": []}')

        registry = _mock_tool_registry("vector_search", "bm25_search", "hybrid_search", "rerank_chunks")
        agent = RAGAgent(tool_registry=registry, llm_provider=mock_llm)

        inp = _make_input(
            task="find info about testing",
            tools_available=["vector_search"],
        )
        result = await agent.execute(inp)
        assert isinstance(result, AgentOutput)
        assert result.status in ("success", "partial", "failed")


# ── Code Agent ─────────────────────────────────────────────────────────────────


class TestCodeAgent:
    @pytest.mark.asyncio
    async def test_execute_returns_agent_output(self):
        from agents.code_agent.agent import CodeAgent

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value='print("hello world")')

        registry = _mock_tool_registry("execute_code", "analyze_code", "generate_code")
        agent = CodeAgent(tool_registry=registry, llm_provider=mock_llm)

        inp = _make_input(
            task="write hello world",
            tools_available=["execute_code"],
        )
        result = await agent.execute(inp)
        assert isinstance(result, AgentOutput)


# ── Mail Agent ─────────────────────────────────────────────────────────────────


class TestMailAgent:
    @pytest.mark.asyncio
    async def test_execute_returns_agent_output(self):
        from agents.mail_agent.agent import MailAgent

        mock_llm = MagicMock()
        # LLM first returns action plan, then formats the final answer
        mock_llm.generate = AsyncMock(side_effect=[
            '{"actions": [{"tool": "search_messages", "args": {"query": "meeting"}}]}',
            '{"summary": "Found 2 messages about meetings.", "details": []}',
        ])

        registry = _mock_tool_registry(
            "send_email", "draft_email", "search_messages",
            "search_drafts", "search_sent_messages",
            "get_message_by_id", "reply_to_message",
        )
        agent = MailAgent(tool_registry=registry, llm_provider=mock_llm)

        inp = _make_input(
            task="search my inbox for messages about the meeting",
            tools_available=["search_messages"],
        )
        result = await agent.execute(inp)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "mail_agent"

    @pytest.mark.asyncio
    async def test_required_tools_listed(self):
        from agents.mail_agent.agent import MailAgent

        registry = _mock_tool_registry()
        agent = MailAgent(tool_registry=registry, llm_provider=MagicMock())
        required = agent.get_required_tools()
        assert "send_email" in required
        assert "search_messages" in required
        assert "reply_to_message" in required
        assert len(required) == 7


# ── Web Search Agent ───────────────────────────────────────────────────────────


class TestWebSearchAgent:
    @pytest.mark.asyncio
    async def test_execute_returns_agent_output(self):
        from agents.web_search_agent.agent import WebSearchAgent

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=[
            # search strategy
            '{"search_query": "latest AI news", "search_type": "basic", "follow_up_urls": false, "reasoning": "simple query"}',
            # final answer synthesis
            '{"answer": "Here are the latest AI developments...", "sources": []}',
        ])

        registry = _mock_tool_registry("web_search", "fetch_url", "web_search_news", "web_search_deep")
        agent = WebSearchAgent(tool_registry=registry, llm_provider=mock_llm)

        inp = _make_input(
            task="what are the latest AI news",
            tools_available=["web_search", "fetch_url"],
        )
        result = await agent.execute(inp)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "web_search_agent"

    @pytest.mark.asyncio
    async def test_required_tools_listed(self):
        from agents.web_search_agent.agent import WebSearchAgent

        registry = _mock_tool_registry()
        agent = WebSearchAgent(tool_registry=registry, llm_provider=MagicMock())
        required = agent.get_required_tools()
        assert "web_search" in required
        assert "fetch_url" in required
