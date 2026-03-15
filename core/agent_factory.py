"""
Centralised agent-instance builder.

Every place that needs the full set of agent objects (startup validation,
/query endpoint, /query/stream endpoint) calls `build_agent_instances`
so the wiring logic lives in exactly one place.
"""

from __future__ import annotations

from typing import Dict

from agents.base_agent import BaseAgent
from agents.code_agent.agent import CodeAgent
from agents.github_agent.agent import GitHubAgent
from agents.mail_agent.agent import MailAgent
from agents.rag_agent.agent import RAGAgent
from agents.web_search_agent.agent import WebSearchAgent
from config.settings import config, model_provider_for, _VALID_MODELS
from tools.registry import ToolRegistry
from utils.llm_providers import get_llm_provider


def build_agent_instances(
    registry: ToolRegistry,
    model_override: str | None = None,
) -> Dict[str, BaseAgent]:
    """
    Construct every agent with its dedicated LLM provider and the
    shared ToolRegistry.

    When *model_override* is a valid model ID, **all** agents use that
    model instead of their per-agent config defaults.
    """
    if model_override and model_override in _VALID_MODELS:
        _provider = model_provider_for(model_override)
        _override_llm = get_llm_provider(_provider, default_model=model_override)
    else:
        _override_llm = None

    def _llm(agent_name: str):
        if _override_llm:
            return _override_llm
        cfg = config.get_agent_model_config(agent_name)
        return get_llm_provider(cfg["provider"], default_model=cfg["model"])

    return {
        "rag_agent": RAGAgent(
            tool_registry=registry,
            llm_provider=_llm("rag_agent"),
        ),
        "code_agent": CodeAgent(
            tool_registry=registry,
            llm_provider=_llm("code_agent"),
        ),
        "mail_agent": MailAgent(
            tool_registry=registry,
            llm_provider=_llm("mail_agent"),
        ),
        "web_search_agent": WebSearchAgent(
            tool_registry=registry,
            llm_provider=_llm("web_search_agent"),
        ),
        "github_agent": GitHubAgent(
            tool_registry=registry,
            llm_provider=_llm("github_agent"),
        ),
    }
