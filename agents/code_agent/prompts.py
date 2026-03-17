"""
Code agent prompts
"""

from __future__ import annotations

from typing import Any, Dict

from utils.prompt_utils import format_user_profile


class CodePrompts:

    @staticmethod
    def _format_persona_block(persona: Dict[str, Any] | None) -> str:
        """Build a persona instruction block for code generation."""
        if not persona or not persona.get("name"):
            return ""
        return (
            f"\n### Persona — CRITICAL\n"
            f"You MUST adopt this persona for ALL generated content:\n"
            f"**Name**: {persona.get('name', '')}\n"
            f"**Style**: {persona.get('description', '')}\n\n"
            f"Every piece of text you generate (PDF content, report text, chart titles, "
            f"data labels, print statements) MUST reflect this persona's:\n"
            f"- Tone and vocabulary\n"
            f"- Personality and attitude\n"
            f"- Way of addressing the user\n"
            f"- Level of formality or casualness\n\n"
            f"Examples:\n"
            f"- If persona is 'Professor': Use academic language, cite sources formally, be educational\n"
            f"- If persona is 'Friend': Be casual, use everyday language, add friendly comments\n"
            f"- If persona is 'Lover': Be warm, affectionate, caring in tone\n"
            f"- If persona is 'Einstein': Be intellectually curious, use analogies, explain with wonder\n"
        )

    @staticmethod
    def _format_dependency_context(dependency_outputs: Dict[str, Any] | None) -> str:
        """Format data from upstream agents for use in code generation."""
        if not dependency_outputs:
            return ""

        context = "\n### Data from Upstream Agents — USE THIS DATA\n"
        context += "The following data was gathered by other agents. Your code should use this information:\n\n"

        for aid, data in dependency_outputs.items():
            agent_type = aid.split("_")[0] if "_" in aid else aid
            data_str = str(data)

            # Provide more context based on agent type
            if "web_search" in aid:
                context += f"**Web Research Results** (`{aid}`):\n"
                context += f"```json\n{data_str[:1500]}\n```\n"
                context += "↳ Use the search results, snippets, and sources to create your content.\n\n"
            elif "rag" in aid:
                context += f"**Document Search Results** (`{aid}`):\n"
                context += f"```json\n{data_str[:1500]}\n```\n"
                context += "↳ Use the document chunks and extracted information in your output.\n\n"
            elif "mail" in aid:
                context += f"**Email Data** (`{aid}`):\n"
                context += f"```json\n{data_str[:1500]}\n```\n"
                context += "↳ Incorporate email content/metadata as needed.\n\n"
            elif "github" in aid:
                context += f"**GitHub Data** (`{aid}`):\n"
                context += f"```json\n{data_str[:1500]}\n```\n"
                context += "↳ Use repository/issue/PR information in your output.\n\n"
            else:
                context += f"**{aid}**:\n"
                context += f"```json\n{data_str[:1200]}\n```\n\n"

        return context

    @staticmethod
    def code_generation_prompt(
        task: str,
        entities: Dict[str, Any],
        dependency_outputs: Dict[str, Any] | None = None,
        long_term_memory: Dict[str, Any] | None = None,
        conversation_history: list | None = None,
        persona: Dict[str, Any] | None = None,
    ) -> str:
        dep_context = CodePrompts._format_dependency_context(dependency_outputs)

        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"

        profile_section = format_user_profile(long_term_memory or {})

        # Extract detail level preference
        detail_level = "balanced"
        if long_term_memory and "preferences" in long_term_memory:
            detail_level = long_term_memory["preferences"].get("detail_level", "balanced")

        conv_section = ""
        if conversation_history:
            conv_section = "\n### Conversation History\n"
            for turn in conversation_history[-10:]:
                role = turn.get("role", "user") if isinstance(turn, dict) else "user"
                content = str(turn.get("content", "") if isinstance(turn, dict) else turn)[:800]
                conv_section += f"  {role}: {content}\n"

        persona_section = CodePrompts._format_persona_block(persona)

        return f"""You are an Expert Developer in a multi-agent RAG system.

### Task
{task}

### Entities
{entity_str}
{dep_context}
{profile_section}
{conv_section}
{persona_section}
### Content Generation Guidelines

**Detail Level**: {detail_level}
- If "concise": Keep content brief, highlight only key points, minimal elaboration
- If "detailed": Provide comprehensive information, explanations, and context
- If "balanced": Mix of key points with some supporting details

**Using Upstream Agent Data**:
- Web search results contain real-time information — extract key facts, quotes, and sources
- Document search results contain relevant excerpts — synthesize and cite them
- Always attribute information to sources when generating reports/documents

### Code Instructions

1. Write **clean, production-quality code** that accomplishes the task.
2. Return ONLY the code — no markdown fences, no explanations.
3. **Handle errors gracefully** — use try/except where appropriate.

4. **CRITICAL — File Output**: When generating files (CSV, PDF, chart, image, report):
   - MUST save to `OUTPUT_DIR` directory (pre-defined variable)
   - Use descriptive filenames — files are automatically captured as artifacts
   - Print confirmation: `print("Saved report.pdf")`

5. **PDF Generation** (reportlab):
   - Use `reportlab.platypus` (SimpleDocTemplate, Paragraph, Spacer, Table) for proper formatting
   - ALWAYS use text wrapping — never let text overflow margins
   - Set margins: 0.75 inch on all sides
   - Structure:
     ```python
     from reportlab.lib.pagesizes import letter
     from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
     from reportlab.lib.units import inch
     from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table

     doc = SimpleDocTemplate(filepath, pagesize=letter,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
     styles = getSampleStyleSheet()
     story = []
     story.append(Paragraph("Title", styles['Title']))
     story.append(Paragraph("Content...", styles['Normal']))
     doc.build(story)
     ```
   - **Apply persona tone** in all text content within the PDF

6. **Chart Generation** (matplotlib):
   - Set figure size: `plt.figure(figsize=(10, 6))`
   - Add labels and title (reflect persona tone if applicable)
   - Use `plt.tight_layout()` before saving
   - Save: `plt.savefig(path, dpi=150, bbox_inches='tight')`

7. **CSV**: Include headers, use UTF-8 encoding

8. **Print results** for non-file tasks in readable format

9. **Available packages**: json, math, statistics, datetime, collections, itertools, re,
   csv, io, matplotlib, pandas, reportlab

10. Parse upstream agent data carefully — extract relevant information for your output

11. **PERSONA REMINDER**: If a persona is specified above, ALL generated text must reflect that persona's style!

### Code"""
