"""
Mail Agent — Gmail integration for searching, composing, and sending emails.

Supports:
  • search_messages / search_sent_messages / search_drafts
  • send_email / draft_email / reply_to_message
  • get_message_by_id (for reading full messages)

The LLM decides what action(s) to take based on the task description.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.mail_agent.prompts import MailPrompts
from config.settings import config
from utils.schemas import AgentInput, AgentOutput
import logging
logger = logging.getLogger("mail_agent")

class MailAgent(BaseAgent):
    def __init__(self, tool_registry, llm_provider):
        super().__init__("mail_agent", tool_registry)
        self.llm = llm_provider
        self.prompts = MailPrompts()

    def get_required_tools(self) -> List[str]:
        return [
            "send_email",
            "draft_email",
            "search_messages",
            "search_drafts",
            "search_sent_messages",
            "get_message_by_id",
            "reply_to_message",
        ]

    async def execute(self, task_config: AgentInput) -> AgentOutput:
        start = time.perf_counter()


        logger.info(f"[MailAgent] Input: {task_config}")
        try:
            plan_prompt = self.prompts.action_plan_prompt(
                task=task_config.task,
                entities=task_config.entities,
                dependency_outputs=task_config.dependency_outputs,
                long_term_memory=task_config.long_term_memory,
            )
            plan = await self.llm.generate(
                prompt=plan_prompt,
                temperature=config.mail_temperature,
                model=config.mail_model,
                output_schema={
                    "action": "search_inbox | search_sent | search_drafts | send | draft | reply | read",
                    "search_query": "string or null",
                    "email": {
                        "to": "string or null",
                        "subject": "string or null",
                        "body": "string or null",
                        "cc": "string or null",
                        "html": "boolean",
                    },
                    "message_id": "string or null  (for reply / read)",
                    "reasoning": "string",
                },
            )

            if not isinstance(plan, dict):
                plan = {"action": "search_inbox", "search_query": task_config.task}

            action = plan.get("action", "search_inbox")
            result_data: Dict[str, Any] = {"action": action}
            status = "success"

            if action == "search_inbox":
                tool_fn = self.get_tool("search_messages")
                query = plan.get("search_query", task_config.task)
                messages = await tool_fn(query=query, max_results=10)
                result_data["messages"] = messages
                result_data["count"] = len(messages)

            elif action == "search_sent":
                tool_fn = self.get_tool("search_sent_messages")
                query = plan.get("search_query", "")
                messages = await tool_fn(query=query, max_results=10)
                result_data["messages"] = messages
                result_data["count"] = len(messages)

            elif action == "search_drafts":
                tool_fn = self.get_tool("search_drafts")
                query = plan.get("search_query", "")
                drafts = await tool_fn(query=query, max_results=10)
                result_data["drafts"] = drafts
                result_data["count"] = len(drafts)

            elif action == "read":
                tool_fn = self.get_tool("get_message_by_id")
                msg_id = plan.get("message_id", "")
                if msg_id:
                    message = await tool_fn(message_id=msg_id)
                    result_data["message"] = message
                else:
                    result_data["error"] = "No message_id provided for read action"
                    status = "partial"

            elif action == "send":
                email = plan.get("email", {})
                if not email.get("to"):
                    result_data["error"] = "No recipient provided"
                    status = "failed"
                else:
                    if not email.get("body") or len(email.get("body", "")) < 20:
                        compose_prompt = self.prompts.compose_email_prompt(
                            task=task_config.task,
                            entities=task_config.entities,
                            dependency_outputs=task_config.dependency_outputs,
                            draft_email=email,
                            long_term_memory=task_config.long_term_memory,
                        )
                        email = await self.llm.generate(
                            prompt=compose_prompt,
                            temperature=config.mail_temperature,
                            model=config.mail_model,
                            output_schema={
                                "to": "string",
                                "subject": "string",
                                "body": "string",
                                "cc": "string or null",
                                "html": "boolean",
                            },
                        )
                        if not isinstance(email, dict):
                            email = plan.get("email", {})

                    tool_fn = self.get_tool("send_email")
                    sent = await tool_fn(
                        to=email.get("to", ""),
                        subject=email.get("subject", ""),
                        body=email.get("body", ""),
                        cc=email.get("cc"),
                        html=email.get("html", False),
                    )
                    result_data["sent"] = sent

            elif action == "draft":
                email = plan.get("email", {})
                compose_prompt = self.prompts.compose_email_prompt(
                    task=task_config.task,
                    entities=task_config.entities,
                    dependency_outputs=task_config.dependency_outputs,
                    draft_email=email,
                    long_term_memory=task_config.long_term_memory,
                )
                email = await self.llm.generate(
                    prompt=compose_prompt,
                    temperature=config.mail_temperature,
                    model=config.mail_model,
                    output_schema={
                        "to": "string",
                        "subject": "string",
                        "body": "string",
                        "cc": "string or null",
                        "html": "boolean",
                    },
                )
                if not isinstance(email, dict):
                    email = plan.get("email", {})

                tool_fn = self.get_tool("draft_email")
                draft = await tool_fn(
                    to=email.get("to", ""),
                    subject=email.get("subject", ""),
                    body=email.get("body", ""),
                    cc=email.get("cc"),
                    html=email.get("html", False),
                )
                result_data["draft"] = draft

            elif action == "reply":
                msg_id = plan.get("message_id")
                email = plan.get("email", {})
                if not msg_id:
                    result_data["error"] = "No message_id for reply"
                    status = "failed"
                else:
                    tool_fn = self.get_tool("reply_to_message")
                    reply = await tool_fn(
                        message_id=msg_id,
                        body=email.get("body", ""),
                        html=email.get("html", False),
                    )
                    result_data["reply"] = reply

            else:
                result_data["error"] = f"Unknown action: {action}"
                status = "failed"

            logger.info(f"[MailAgent] Output: {result_data}")
            return AgentOutput(
                agent_id=task_config.agent_id,
                agent_name=self.agent_name,
                task_description=task_config.task,
                status=status,
                task_done=status == "success",
                result=result_data.get("sent") or result_data.get("draft") or result_data.get("reply") or result_data.get("messages") or result_data.get("drafts") or result_data.get("message"),
                data=result_data,
                confidence_score=0.9 if status == "success" else 0.4,
                resource_usage={
                    "time_taken_ms": int((time.perf_counter() - start) * 1000),
                },
                depends_on=list(task_config.dependency_outputs.keys()),
            )

        except Exception:
            logger.exception(f"[MailAgent] Error for input: {task_config}")
            raise

