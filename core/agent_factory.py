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
from config.settings import config
from tools.registry import ToolRegistry
from utils.llm_providers import get_llm_provider


def build_agent_instances(registry: ToolRegistry) -> Dict[str, BaseAgent]:
    """
    Construct every agent with its dedicated LLM provider and the
    shared ToolRegistry.

    LLM providers are cached internally by ``get_llm_provider``, so
    calling this multiple times is cheap.
    """
    rag_cfg = config.get_agent_model_config("rag_agent")
    code_cfg = config.get_agent_model_config("code_agent")
    mail_cfg = config.get_agent_model_config("mail_agent")
    web_cfg = config.get_agent_model_config("web_search_agent")
    github_cfg = config.get_agent_model_config("github_agent")

    return {
        "rag_agent": RAGAgent(
            tool_registry=registry,
            llm_provider=get_llm_provider(rag_cfg["provider"], default_model=rag_cfg["model"]),
        ),
        "code_agent": CodeAgent(
            tool_registry=registry,
            llm_provider=get_llm_provider(code_cfg["provider"], default_model=code_cfg["model"]),
        ),
        "mail_agent": MailAgent(
            tool_registry=registry,
            llm_provider=get_llm_provider(mail_cfg["provider"], default_model=mail_cfg["model"]),
        ),
        "web_search_agent": WebSearchAgent(
            tool_registry=registry,
            llm_provider=get_llm_provider(web_cfg["provider"], default_model=web_cfg["model"]),
        ),
        "github_agent": GitHubAgent(
            tool_registry=registry,
            llm_provider=get_llm_provider(github_cfg["provider"], default_model=github_cfg["model"]),
        ),
    }
