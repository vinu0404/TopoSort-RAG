"""
Input validation and query blocklist checking.

Provides a FastAPI-compatible ``validate_query_input`` that raises
``HTTPException(400)`` when the incoming query is empty, too long,
or matches known jailbreak / prompt-extraction patterns.
"""

from __future__ import annotations

import logging
import re
from typing import List

from fastapi import HTTPException

logger = logging.getLogger("security.validators")

# ── Compiled blocklist patterns ───────────────────────────────────────────
_BLOCKLIST_DEFINITIONS: list[tuple[str, str]] = [
    # System-prompt extraction
    ("system_prompt_extract",
     r"(what|show|tell|reveal|print|display|give)\s*(me\s*)?(your|the)?\s*(system|initial|original|hidden)\s*(prompt|instruction|message)s?"),
    ("repeat_above",
     r"repeat\s+(everything|all|text)\s+(above|before|previous)"),
    ("output_prompt",
     r"output\s+(your\s+)?(system|initial|hidden)\s*(prompt|instruction)s?"),

    # Role manipulation / jailbreak
    ("developer_mode",
     r"(enable|activate|turn\s+on)\s+(developer|dev|admin|god)\s*(mode)?"),
    ("dan_jailbreak",
     r"\b(DAN|do\s+anything\s+now)\b"),
    ("jailbreak_keyword",
     r"\bjailbreak\b"),

    # Model token manipulation
    ("token_endoftext",
     r"<\|endoftext\|>"),
    ("token_im",
     r"<\|im_(start|end)\|>"),
    ("token_bracket",
     r"\[TOKEN[S]?\]"),
]

_BLOCKLIST_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pat, re.IGNORECASE))
    for name, pat in _BLOCKLIST_DEFINITIONS
]


def check_query_blocklist(query: str) -> List[str]:
    """
    Check *query* against compiled blocklist patterns.

    Returns a list of matched pattern *names* (empty if clean).
    """
    return [
        name
        for name, pattern in _BLOCKLIST_COMPILED
        if pattern.search(query)
    ]


def validate_query_input(
    query: str,
    *,
    max_length: int = 50_000,
) -> None:
    """
    Validate an incoming user query.

    Raises :class:`HTTPException` with status 400 when:
    * the query is empty or whitespace-only,
    * exceeds *max_length*, or
    * matches a blocklist pattern.

    On blocklist match we log the details but **do not** reveal which
    pattern was hit in the client-facing error message.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": "Query cannot be empty"},
        )

    if len(query) > max_length:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": "Query exceeds maximum length"},
        )

    matches = check_query_blocklist(query)
    if matches:
        logger.warning("Query blocklist hit: patterns=%s query=%.200s", matches, query)
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": "Query contains invalid content"},
        )
