"""
GitHub Agent — repo info, issues, PRs, repo creation, PR creation.

Token pass-through pattern:
  • get_active_token() fetches a fresh per-user token on every request.
  • Token is passed directly to each tool function (no ContextVar needed).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.github_agent.prompts import GitHubPrompts
from config.settings import config
from utils.schemas import AgentInput, AgentOutput
# ── Fresh token every request ────────
from connectors.token_manager import get_active_token
import logging

logger = logging.getLogger("github_agent")


class GitHubAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("github_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = GitHubPrompts()

    def get_required_tools(self) -> List[str]:
        return [
            "list_user_repos",
            "get_repo_info",
            "list_repo_issues",
            "list_pull_requests",
            "create_repo",
            "create_pull_request",
        ]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()
        user_id = task_config.metadata.get("user_id", "")



        token = await get_active_token(user_id, "github")
        if not token:
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                status="failed",
                task_done=False,
                result="GitHub not connected. Please connect your GitHub account via Settings → Connections.",
                data={"error": "not_connected"},
            )

        try:
            # HITL-aware effective task (handles enhance/override)
            effective_task = await self._effective_task(task_config)

            plan = await self.llm.generate(
                prompt=self.prompts.action_prompt(
                    task=effective_task,
                    entities=task_config.entities,
                    dependency_outputs=task_config.dependency_outputs,
                    long_term_memory=task_config.long_term_memory,
                    conversation_history=task_config.conversation_history,
                ),
                temperature=config.get_agent_model_config("github_agent")["temperature"],
                model=config.get_agent_model_config("github_agent")["model"],
                output_schema={
                    "action": "list_repos | repo_info | list_issues | list_prs | create_repo | create_pr",
                    "params": "dict of tool parameters",
                    "reasoning": "string",
                },
            )

            if not isinstance(plan, dict):
                plan = {"action": "repo_info", "params": {}}

            action = plan.get("action", "repo_info")
            params = plan.get("params", {})
            result_data: Dict[str, Any] = {"action": action}
            status = "success"

            # Token is passed directly to each tool — no ContextVar needed
            if action == "list_repos":
                tool_fn = self.get_tool("list_user_repos")
                result_data["repos"] = await tool_fn(
                    token=token,
                    sort=params.get("sort", "updated"),
                    limit=params.get("limit", 10),
                )

            elif action == "repo_info":
                tool_fn = self.get_tool("get_repo_info")
                result_data["repo"] = await tool_fn(
                    token=token,
                    owner=params.get("owner", ""),
                    repo=params.get("repo", ""),
                )

            elif action == "list_issues":
                tool_fn = self.get_tool("list_repo_issues")
                result_data["issues"] = await tool_fn(
                    token=token,
                    owner=params.get("owner", ""),
                    repo=params.get("repo", ""),
                    state=params.get("state", "open"),
                    limit=params.get("limit", 10),
                )

            elif action == "list_prs":
                tool_fn = self.get_tool("list_pull_requests")
                result_data["pull_requests"] = await tool_fn(
                    token=token,
                    owner=params.get("owner", ""),
                    repo=params.get("repo", ""),
                    state=params.get("state", "open"),
                    limit=params.get("limit", 10),
                )

            elif action == "create_repo":
                tool_fn = self.get_tool("create_repo")
                result_data["created"] = await tool_fn(
                    token=token,
                    name=params.get("name", ""),
                    description=params.get("description", ""),
                    private=params.get("private", False),
                )

            elif action == "create_pr":
                tool_fn = self.get_tool("create_pull_request")
                result_data["pull_request"] = await tool_fn(
                    token=token,
                    owner=params.get("owner", ""),
                    repo=params.get("repo", ""),
                    title=params.get("title", ""),
                    head=params.get("head", ""),
                    base=params.get("base", "main"),
                    body=params.get("body", ""),
                    draft=params.get("draft", False),
                )

            else:
                result_data["error"] = f"Unknown action: {action}"
                status = "failed"

            logger.info("[GitHubAgent] Output: %s", result_data)
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=effective_task,
                status=status,
                task_done=status == "success",
                result=(
                    result_data.get("created")
                    or result_data.get("pull_request")
                    or result_data.get("repo")
                    or result_data.get("repos")
                    or result_data.get("issues")
                    or result_data.get("pull_requests")
                ),
                data=result_data,
                confidence_score=0.85 if status == "success" else 0.4,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                },
                depends_on=list(task_config.dependency_outputs.keys()),
            )

        except Exception:
            logger.exception("[GitHubAgent] Error for input: %s", task_config)
            raise
