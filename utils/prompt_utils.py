"""
Shared prompt helpers
"""

from __future__ import annotations

from typing import Any, Dict


def format_user_profile(
    long_term_memory: Dict[str, Any],
    *,
    header: str = "User Profile",
) -> str:
    """
    Build a markdown User Profile section from a ``long_term_memory`` dict.

    Parameters
    ----------
    long_term_memory : dict with ``critical`` and ``preferences`` sub-dicts.
    header           : the ``### <header>`` text (override per-agent if needed).

    Returns
    -------
    A formatted string like::

        ### User Profile
        Name: Alice
        Company: Acme Corp
        Role: Data Engineer
        Projects: Alpha, Beta
        Preferred tone: casual | Detail: concise | Language: en

    or an empty string when there is nothing to render.
    """
    if not long_term_memory:
        return ""

    critical = long_term_memory.get("critical", {})
    prefs = long_term_memory.get("preferences", {})

    parts: list[str] = []

    if critical.get("user_name"):
        parts.append(f"Name: {critical['user_name']}")
    if critical.get("company_name"):
        parts.append(f"Company: {critical['company_name']}")
    if critical.get("job_title"):
        parts.append(f"Role: {critical['job_title']}")
    if critical.get("recent_projects"):
        projects = critical["recent_projects"]
        if isinstance(projects, list):
            parts.append(f"Projects: {', '.join(projects)}")
        else:
            parts.append(f"Projects: {projects}")

    tone = prefs.get("tone", "professional")
    detail = prefs.get("detail_level", "concise")
    lang = prefs.get("language", "en")
    parts.append(f"Preferred tone: {tone} | Detail: {detail} | Language: {lang}")

    if not parts:
        return ""

    return f"\n### {header}\n" + "\n".join(parts) + "\n"
