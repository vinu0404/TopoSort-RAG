"""Code Agent — generates and executes code."""

from __future__ import annotations

import time
from typing import  List

from agents.base_agent import BaseAgent
from agents.code_agent.prompts import CodePrompts
from config.settings import config
from utils.schemas import AgentInput, AgentOutput
import logging
logger = logging.getLogger("code_agent")

class CodeAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("code_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = CodePrompts()

    def get_required_tools(self) -> List[str]:
        return ["execute_code", "code_linter"]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()
        logger.info(f"[CodeAgent] Input: {task_config}")
        try:
            execute_code = self.get_tool("execute_code")
            linter = self.get_tool("code_linter")
            prompt = self.prompts.code_generation_prompt(
                task=task_config.task,
                entities=task_config.entities,
                dependency_outputs=task_config.dependency_outputs,
                long_term_memory=task_config.long_term_memory,
            )
            code: str = await self.llm.generate(
                prompt=prompt,
                temperature=config.code_temperature,
                model=config.code_model,
            )
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            issues = await linter(code)
            if any(i["severity"] == "error" for i in issues):
                return AgentOutput(
                    agent_id=task_config.agent_id,
                    agent_name=self.agent_name,
                    task_description=task_config.task,
                    task_done=False,
                    error="Generated code has syntax errors",
                    partial_data={"code": code, "lint_issues": issues},
                    depends_on=list(task_config.dependency_outputs.keys()),
                )
            result = await execute_code(code)

            logger.info(f"[CodeAgent] Output: {result}")
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                task_done=result.get("exit_code") == 0,
                data={
                    "code": code,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "exit_code": result.get("exit_code"),
                },
                confidence_score=1.0 if result.get("exit_code") == 0 else 0.3,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                },
                depends_on=list(task_config.dependency_outputs.keys()),
            )

        except Exception:
            logger.exception(f"[CodeAgent] Error for input: {task_config}")
            raise
