"""
GitHub agent prompts — action planning for repo management, issues, and PRs.
"""

from __future__ import annotations

from typing import Any, Dict

from security.sanitization import sanitize_user_input
from security.delimiters import wrap_task, wrap_conversation_history, DELIMITER_SYSTEM_PROMPT
from utils.prompt_utils import format_user_profile


class GitHubPrompts:

    @staticmethod
    def action_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:500]}\n"

        profile = format_user_profile(long_term_memory or {})

        history_context = ""
        if conversation_history:
            conv_lines = ""
            for msg in conversation_history[-10:]:
                role = msg.get("role", "user")
                text = str(msg.get("content", ""))[:800]
                conv_lines += f"- {role}: {text}\n"
            history_context = "\n### Recent conversation\n" + wrap_conversation_history(conv_lines)

        entities_str = ""
        if entities:
            entities_str = "\n### Extracted entities\n"
            for k, v in entities.items():
                entities_str += f"- {sanitize_user_input(str(k)).text}: {sanitize_user_input(str(v)).text}\n"

        safe_task = sanitize_user_input(task).text

        return f"""{DELIMITER_SYSTEM_PROMPT}

You are a GitHub Integration Agent in a multi-agent system.

{wrap_task(safe_task)}
{dep_context}
{profile}
{history_context}
{entities_str}

### Available Actions
- **list_repos** — list the authenticated user's repositories (no owner/repo needed)
- **repo_info**   — get metadata for a specific repository (owner, repo required)
- **list_issues** — list issues in a repository (owner, repo required)
- **list_prs**    — list pull requests in a repository (owner, repo required)
- **create_repo** — create a new repository (name required) ⚠️ requires approval
- **create_pr**   — create a pull request (owner, repo, title, head, base required) ⚠️ requires approval

### Instructions
1. Parse the user's request to determine which action to take.
2. Extract the required parameters (owner, repo, name, etc.) from the task and entities.
3. If the user mentions a repo like "owner/repo", split it into owner and repo.
4. If the user asks about their repos without specifying one, use **list_repos**.
5. For create operations, include a clear description/body.
5. Return your plan as JSON with the action and params.

### Output format
Return JSON:
```json
{{
  "action": "list_repos | repo_info | list_issues | list_prs | create_repo | create_pr",
  "params": {{...}},
  "reasoning": "why this action"
}}
```"""
