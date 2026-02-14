"""
GitHub agent prompts — action planning for repo management, issues, and PRs.
"""

from __future__ import annotations

from typing import Any, Dict

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
            history_context = "\n### Recent conversation\n"
            for msg in conversation_history[-4:]:
                role = msg.get("role", "user")
                text = str(msg.get("content", ""))[:300]
                history_context += f"- {role}: {text}\n"

        entities_str = ""
        if entities:
            entities_str = "\n### Extracted entities\n"
            for k, v in entities.items():
                entities_str += f"- {k}: {v}\n"

        return f"""You are a GitHub Integration Agent in a multi-agent system.

### Task
{task}
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
