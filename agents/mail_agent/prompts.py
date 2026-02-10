"""
Mail agent prompts — production-quality prompts for the Gmail-integrated agent.

Covers:
  • Action planning (decide search vs send vs draft vs reply)
  • Email composition (polished, professional)
  • Result summarisation
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


from utils.prompt_utils import format_user_profile


class MailPrompts:

    @staticmethod
    def action_plan_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:400]}\n"

        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"

        profile_section = format_user_profile(long_term_memory or {}, header="User Profile (use for sign-off name and email tone)")

        return f"""You are the Gmail Action Planner for a multi-agent RAG system.

### Your Task
Determine what Gmail action to take for the user's request and provide the needed parameters.

### User Request
{task}

### Extracted Entities
{entity_str}
{dep_context}
{profile_section}

### Available Actions
| Action           | When to use                                         |
|------------------|-----------------------------------------------------|
| search_inbox     | User wants to find/read emails in their inbox       |
| search_sent      | User asks about emails they sent                    |
| search_drafts    | User asks about draft emails                        |
| read             | User wants the full content of a specific message   |
| send             | User explicitly asks to send an email               |
| draft            | User wants to compose but NOT send, or you're unsure|
| reply            | User wants to reply to a specific email thread      |

### Gmail search syntax examples (for search_query)
- `from:john@example.com` — emails from a specific person
- `subject:meeting` — emails with "meeting" in the subject
- `after:2025/01/01 before:2025/02/01` — date range
- `has:attachment` — emails with attachments
- `is:unread` — unread only
- `label:important` — specific label
- Combine: `from:boss@company.com subject:quarterly after:2025/01/01`

### Instructions
- For **search** actions, generate a precise Gmail search query.
- For **send/draft/reply**, fill in the email fields.
- Default to **draft** instead of **send** if the user's intent to actually send isn't explicit.
- If the user says "reply to the email from X about Y", use action=reply and search for the message first.

### Output (JSON)
Return EXACTLY this JSON:
{{
    "action": "search_inbox | search_sent | search_drafts | send | draft | reply | read",
    "search_query": "<gmail search query or null>",
    "email": {{
        "to": "<recipient or null>",
        "subject": "<subject or null>",
        "body": "<body text or null>",
        "cc": "<cc or null>",
        "html": false
    }},
    "message_id": "<message ID for reply/read, or null>",
    "reasoning": "<brief explanation of your choice>"
}}"""

    @staticmethod
    def compose_email_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        draft_email: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Context from other agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"- **{aid}**: {str(data)[:500]}\n"

        existing = ""
        if draft_email:
            existing = f"\n### Existing draft to polish\n```json\n{json.dumps(draft_email, indent=2)}\n```\n"

        profile_section = format_user_profile(long_term_memory or {}, header="User Profile (use for sign-off name and email tone)")

        return f"""You are a Professional Email Composer for a multi-agent RAG system.

### Task
{task}

### Entities
{entities}
{dep_context}
{existing}
{profile_section}
### Instructions
1. Write a clear, professional, and appropriately toned email.
2. **Personalise using User Profile**: Use the user's name for the sign-off, match their
   preferred tone (formal/casual/friendly), and write in their preferred language.
3. Include a proper greeting and sign-off.
4. If you received data from other agents, incorporate it naturally into the email body.
5. Keep the email focused and actionable — avoid unnecessary filler.
6. If no recipient is specified, leave `to` as an empty string.

### Output (JSON)
Return EXACTLY:
{{
    "to": "<email address>",
    "subject": "<clear, descriptive subject line>",
    "body": "<complete email body with greeting and sign-off>",
    "cc": "<cc address or null>",
    "html": false
}}"""

    @staticmethod
    def summarise_result_prompt(
        task: str,
        action: str,
        data: Dict[str, Any],
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        # Truncate data for prompt
        data_str = json.dumps(data, default=str)
        if len(data_str) > 3000:
            data_str = data_str[:3000] + "... (truncated)"

        prefs = (long_term_memory or {}).get("preferences", {})
        tone = prefs.get("tone", "professional")
        detail = prefs.get("detail_level", "concise")
        lang = prefs.get("language", "en")

        return f"""You are a concise assistant summarising the outcome of a Gmail action.

### Original User Request
{task}

### Action Taken
{action}

### Raw Result
{data_str}

### Personalisation
Tone: {tone} | Detail level: {detail} | Language: {lang}

### Instructions
1. Provide a clear, human-readable summary of what happened.
2. For **search** results: list the most relevant messages (sender, subject, date, snippet).
   Mention the total count.
3. For **send/draft**: confirm what was sent/drafted, to whom, with what subject.
4. For **reply**: confirm the reply was sent and to which thread.
5. For **errors**: explain what went wrong in plain language.
6. Keep it concise — 2-5 sentences max, plus a bullet list if there are multiple items.

### Summary"""
