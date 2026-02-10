"""
Code agent prompts
"""

from __future__ import annotations

from typing import Any, Dict

from utils.prompt_utils import format_user_profile


class CodePrompts:

    @staticmethod
    def code_generation_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
    ) -> str:
        dep_context = ""
        if dependency_outputs:
            dep_context = "\n### Data from upstream agents\n"
            for aid, data in dependency_outputs.items():
                dep_context += f"**{aid}**:\n```\n{str(data)[:800]}\n```\n\n"

        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"

        profile_section = format_user_profile(long_term_memory or {})

        return f"""You are an Expert in Many programming languages Developer in a multi-agent RAG system.

### Task
{task}

### Entities
{entity_str}
{dep_context}
{profile_section}
### Instructions
1. Write **clean, production-quality code** that accomplishes the task.
2. Return ONLY the code — no markdown fences, no explanations, no comments about what the code does.
3. **Handle errors gracefully** — use try/except where appropriate.
4. **Print results clearly** — use `print()` to output the final result in a human-readable format.
   If the result is structured data, print it as formatted JSON.
5. Available standard library + common packages: `json`, `math`, `statistics`, `datetime`,
   `collections`, `itertools`, `re`, `csv`, `io`.
6. If working with data from other agents, parse it cleanly — don't assume exact formats.
7. For data analysis:
   - Use descriptive variable names.
   - Print intermediate results if they help explain the computation.
   - Format numbers appropriately (rounding, units, percentages).
8. Keep the code concise but correct — prefer readability over cleverness.
9. Always include a `if __name__ == "__main__"` guard or just top-level code that runs directly FOR EASY EXECUTION.
10. Personalise the code style and print formatting based on the user's preferences in the User Profile (e.g. concise vs detailed output, preferred language for print statements).
11. **Personalise**: If the user prefers detailed output, include verbose print statements and
    explanatory comments. If concise, keep output minimal. Print in the user's preferred language
    if one is specified in the User Profile or in query.

### Code"""
