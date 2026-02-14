"""
Abstract base class for every agent in the system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List

from config.settings import config
from utils.llm_providers import get_llm_provider
from utils.schemas import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Every agent inherits from this and implements:
      • execute()           — the core logic
      • get_required_tools() — used for startup validation
    """

    def __init__(self, agent_name: str, tool_registry):
        self.agent_name = agent_name
        self.tool_registry = tool_registry
    @abstractmethod
    async def execute(self, task_config: AgentInput) -> AgentOutput:
        """
        Run the agent's core logic.

        Must NOT handle retries — that is done by `execute_with_retry`.
        """
        ...

    @abstractmethod
    def get_required_tools(self) -> List[str]:
        """Return the tool names this agent requires (checked at startup)."""

    def get_tool(self, tool_name: str):
        """Proxy to ToolRegistry with built-in permission check."""
        return self.tool_registry.get_tool(tool_name, self.agent_name)
    



    # ── HITL intent classification prompt ────────────────────────────────
    _HITL_CLASSIFY_PROMPT = """You are classifying the intent of a user's HITL (Human-in-the-Loop) instruction.

Original task assigned to an AI agent:
"{original_task}"

User's additional instruction provided during approval:
"{instructions}"

Classify the user's instruction as ONE of:
- "enhance"  — the user wants the agent to do the ORIGINAL task AND ALSO incorporate the new instruction (adds extra scope).
- "override" — the user wants the agent to IGNORE the original task and do something DIFFERENT instead.

Examples:
  Original: "Search for AI trends"  |  Instruction: "also include GPU market growth"  → enhance
  Original: "Search for AI trends"  |  Instruction: "search about cricket instead"   → override
  Original: "Send email to alice"   |  Instruction: "also CC bob@example.com"         → enhance
  Original: "Run the sorting code"  |  Instruction: "run a fibonacci script instead"  → override

Respond ONLY with valid JSON."""

    @staticmethod
    async def _classify_hitl_intent(original_task: str, instructions: str) -> str:
        """
        Use a cheap LLM to classify whether the HITL instruction
        is an *enhancement* (add to original) or an *override* (replace original).

        Returns "enhance" or "override".
        """
        try:
            llm = get_llm_provider(
                config.hitl_classifier_provider,
                default_model=config.hitl_classifier_model,
            )
            result = await llm.generate(
                prompt=BaseAgent._HITL_CLASSIFY_PROMPT.format(
                    original_task=original_task,
                    instructions=instructions,
                ),
                temperature=0.0,
                model=config.hitl_classifier_model,
                max_tokens=150,
                output_schema={
                    "intent": "enhance | override",
                    "reasoning": "string",
                },
            )
            if isinstance(result, dict):
                intent = result.get("intent", "enhance").strip().lower()
            else:
                parsed = json.loads(result)
                intent = parsed.get("intent", "enhance").strip().lower()

            if intent not in ("enhance", "override"):
                logger.warning(f"[HITL] Unexpected intent '{intent}', defaulting to 'enhance'")
                intent = "enhance"

            logger.info(f"[HITL] Classified instruction intent as '{intent}'")
            return intent

        except Exception as exc:
            logger.warning(f"[HITL] Intent classification failed ({exc}), defaulting to 'enhance'")
            return "enhance"



    async def _effective_task(self, task_config: AgentInput) -> str:
        """
        Return the task string to feed into prompts.

        Uses an LLM classifier to decide whether HITL instructions
        should **enhance** the original task (do both) or **override** it
        (replace it entirely).
        """
        if not (
            task_config.hitl_context
            and task_config.hitl_context.approved
            and task_config.hitl_context.instructions
        ):
            return task_config.task

        instructions = task_config.hitl_context.instructions
        intent = await self._classify_hitl_intent(task_config.task, instructions)

        if intent == "enhance":
            return (
                f"Complete BOTH the original task AND the additional user instructions below.\n\n"
                f"### Original Task\n"
                f"{task_config.task}\n\n"
                f"### User Enhanced Instructions\n"
                f"{instructions}"
            )
        else:  # override
            return (
                f"IMPORTANT — The user wants you to follow these override instructions "
                f"INSTEAD of the original task.\n\n"
                f"### User Override Instructions\n"
                f"{instructions}\n\n"
                f"### Original Task (for reference only)\n"
                f"{task_config.task}"
            )


    async def execute_with_retry(self, task_config: AgentInput) -> AgentOutput:
        """
        Run `self.execute()` with exponential-backoff retries.
        """
        retry_cfg = task_config.retry_config
        max_retries: int = retry_cfg.get("max_retries", 2)
        timeout: float = float(retry_cfg.get("timeout", 30))
        backoff_mul: float = retry_cfg.get("backoff_multiplier", 2.0)

        start = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                result: AgentOutput = await asyncio.wait_for(
                    self.execute(task_config),
                    timeout=timeout,
                )
                result.execution_metadata["attempt_number"] = attempt + 1
                result.resource_usage["time_taken_ms"] = int(
                    (time.perf_counter() - start) * 1000
                )
                return result

            except asyncio.TimeoutError:
                logger.warning(
                    "%s attempt %d/%d timed out (%.1fs)",
                    task_config.agent_id,
                    attempt + 1,
                    max_retries + 1,
                    timeout,
                )
                if attempt < max_retries:
                    timeout *= backoff_mul
                    await asyncio.sleep(2**attempt)
                else:
                    return self._create_timeout_response(task_config, attempt + 1, start)

            except Exception as exc:
                logger.exception(
                    "%s attempt %d/%d raised %s",
                    task_config.agent_id,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)
                else:
                    return self._create_error_response(task_config, str(exc), attempt + 1, start)

        return self._create_error_response(task_config, "Unknown failure", max_retries + 1, start)

    def _create_timeout_response(
        self, task_config: AgentInput, attempts: int, start: float
    ) -> AgentOutput:
        return AgentOutput(
            agent_id=task_config.agent_id,
            agent_name=self.agent_name,
            task_description=task_config.task,
            task_done=False,
            error="timeout",
            execution_metadata={
                "attempt_number": attempts,
                "timeout_used": task_config.retry_config.get("timeout", 30),
                "warnings": ["All retry attempts timed out"],
            },
            resource_usage={
                "time_taken_ms": int((time.perf_counter() - start) * 1000),
            },
            depends_on=list(task_config.dependency_outputs.keys()),
        )

    def _create_error_response(
        self, task_config: AgentInput, error_msg: str, attempts: int, start: float
    ) -> AgentOutput:
        return AgentOutput(
            agent_id=task_config.agent_id,
            agent_name=self.agent_name,
            task_description=task_config.task,
            task_done=False,
            error=error_msg,
            execution_metadata={
                "attempt_number": attempts,
                "warnings": [f"All {attempts} attempts failed"],
            },
            resource_usage={
                "time_taken_ms": int((time.perf_counter() - start) * 1000),
            },
            depends_on=list(task_config.dependency_outputs.keys()),
        )
