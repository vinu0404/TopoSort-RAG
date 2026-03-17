"""
Input sanitization utilities for prompt-injection defense.

Three public functions — each returns a ``SanitizationResult`` with the
sanitized text, a boolean indicating whether the text was modified,
a list of detected injection pattern names, and a numeric risk score (0‑1).
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("security.sanitization")

# ── Characters used to manipulate prompt structure ────────────────────────
_PROMPT_ESCAPE_MAP: dict[str, str] = {
    "```": "[CODE_FENCE]",
    "###": "[SECTION]",
    "---": "[DIVIDER]",
    "<<<": "[ANGLE_L]",
    ">>>": "[ANGLE_R]",
    "\x00": "",           # Null bytes
}

# ── Regex patterns that signal prompt-injection attempts ──────────────────
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions",    re.compile(r"ignore\s+(previous|above|all|prior)\s+(instructions?|prompts?|rules?)", re.I)),
    ("disregard_instructions", re.compile(r"disregard\s+(previous|above|all|prior)\s+(instructions?|prompts?|rules?)", re.I)),
    ("forget_instructions",    re.compile(r"forget\s+(previous|above|all|prior)\s+(instructions?|prompts?|rules?)", re.I)),
    ("role_override",          re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I)),
    ("new_instructions",       re.compile(r"new\s+instructions?\s*:", re.I)),
    ("role_injection_system",  re.compile(r"(?:^|\n)\s*system\s*:", re.I)),
    ("role_injection_asst",    re.compile(r"(?:^|\n)\s*assistant\s*:", re.I)),
    ("role_injection_human",   re.compile(r"(?:^|\n)\s*human\s*:", re.I)),
    ("token_pipe",             re.compile(r"<\|.*?\|>", re.I)),
    ("inst_tag",               re.compile(r"\[/?INST\]", re.I)),
    ("sys_tag",                re.compile(r"<<?/?SYS>>?", re.I)),
    ("do_anything_now",        re.compile(r"\b(?:DAN|do\s+anything\s+now)\b", re.I)),
    ("developer_mode",         re.compile(r"(?:enable|activate|turn\s+on)\s+(?:developer|dev|admin|god)\s*mode", re.I)),
]


@dataclass
class SanitizationResult:
    """Outcome of a sanitization pass."""

    text: str
    was_modified: bool = False
    detected_patterns: List[str] = field(default_factory=list)
    risk_score: float = 0.0


# ─── Public API ───────────────────────────────────────────────────────────

def sanitize_user_input(
    text: str,
    *,
    escape_markdown: bool = True,
    detect_injection: bool = True,
    max_length: int = 50_000,
) -> SanitizationResult:
    """
    Sanitize user-provided text before embedding it in a prompt.

    * Truncates to *max_length*.
    * HTML-encodes angle-bracket characters.
    * Replaces markdown control characters (``###``, fences, etc.).
    * Scans for known injection patterns and computes a risk score.
    """
    if not text:
        return SanitizationResult("")

    original = text
    detected: list[str] = []
    risk = 0.0

    # Length guard
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
        risk += 0.1

    # Neutralise HTML / angle brackets
    text = html.escape(text, quote=False)

    # Escape markdown prompt-control characters
    if escape_markdown:
        for char, replacement in _PROMPT_ESCAPE_MAP.items():
            if char in text:
                text = text.replace(char, replacement)

    # Detect injection patterns
    if detect_injection:
        for name, pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                detected.append(name)
                risk += 0.2

    # Collapse excessive whitespace (can hide injection tokens)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r" {10,}", "    ", text)

    risk = min(risk, 1.0)

    if detected:
        logger.warning(
            "Injection patterns detected (risk=%.1f): %s",
            risk,
            detected,
        )

    return SanitizationResult(
        text=text,
        was_modified=(text != original),
        detected_patterns=detected,
        risk_score=risk,
    )


def sanitize_document_chunk(
    text: str,
    source: str = "unknown",
    *,
    max_length: int = 8_000,
) -> SanitizationResult:
    """
    Sanitize a document chunk retrieved via RAG.

    Delegates to :func:`sanitize_user_input` and appends a source label
    when the content appears risky.
    """
    result = sanitize_user_input(
        text,
        escape_markdown=True,
        detect_injection=True,
        max_length=max_length,
    )

    if result.risk_score > 0.5:
        result.text = (
            f"[DOCUMENT CONTENT — source: {source}]\n"
            f"{result.text}\n"
            f"[END DOCUMENT]"
        )
        result.was_modified = True

    return result


def sanitize_web_content(
    text: str,
    url: str = "unknown",
    *,
    max_length: int = 10_000,
) -> SanitizationResult:
    """
    Sanitize web-scraped content.

    Always wraps the content in source labels because external web pages
    are the highest-risk vector for injection.
    """
    result = sanitize_user_input(
        text,
        escape_markdown=True,
        detect_injection=True,
        max_length=max_length,
    )

    result.text = (
        f"[WEB CONTENT from {url}]\n"
        f"{result.text}\n"
        f"[END WEB CONTENT]"
    )
    result.was_modified = True
    result.risk_score = min(result.risk_score + 0.1, 1.0)

    return result
