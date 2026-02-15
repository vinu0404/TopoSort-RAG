"""
Gmail agent tools — production per-user OAuth via connectors module.

Architecture:
  • MailAgent.execute() calls ``prepare_gmail_service(user_id)`` once per request.
  • That fetches a fresh token via ``connectors.token_manager.get_active_token()``
    (which auto-refreshes expired tokens) and builds a ``googleapiclient`` service.
  • The service is stored in a ``ContextVar`` — automatically scoped to the
    current async task and garbage-collected when the request ends.
  • Every tool function reads the service from the ContextVar.
  • All sync ``googleapiclient`` calls are offloaded to a thread via
    ``asyncio.to_thread()`` so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from config.settings import config
from tools import tool

logger = logging.getLogger(__name__)


# ── Request-scoped context variables ─────────────────────────────────────
# Set once by prepare_gmail_service(), read by every tool function.
# Automatically cleaned up when the async task (request) finishes.

current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default=""
)

_current_gmail_service: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_current_gmail_service", default=None
)

_current_sender_email: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_sender_email", default=""
)


# ── Service lifecycle ────────────────────────────────────────────────────


async def prepare_gmail_service(user_id: str) -> None:
    """
    Fetch a fresh OAuth token from the connectors module, build a Gmail
    API service, and store it in request-scoped ContextVars.

    Must be called **once** at the start of ``MailAgent.execute()``.

    Raises
    ------
    RuntimeError
        If the user has no active Gmail connection.
    """
    from connectors.token_manager import get_active_token, get_user_connections

    token = await get_active_token(user_id, "gmail")
    if not token:
        raise RuntimeError(
            "Gmail not connected. Please connect your Gmail account "
            "via Settings → Connections before using mail features."
        )

    # Build the service in a thread (googleapiclient discovery does I/O)
    creds = Credentials(token=token)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)

    # Resolve sender email from the connection metadata
    connections = await get_user_connections(user_id)
    sender_email = ""
    for conn in connections:
        if conn["provider"] == "gmail" and conn["status"] == "active":
            sender_email = conn.get("provider_meta", {}).get("email", "")
            if not sender_email:
                sender_email = conn.get("account_label", "")
            break

    current_user_id.set(user_id)
    _current_gmail_service.set(service)
    _current_sender_email.set(sender_email or config.gmail_sender_email)

    logger.debug("Gmail service ready for user=%s sender=%s", user_id, sender_email)


def _get_service():
    """
    Return the request-scoped Gmail API service.

    Raises
    ------
    RuntimeError
        If ``prepare_gmail_service()`` was not called for this request.
    """
    service = _current_gmail_service.get(None)
    if service is None:
        raise RuntimeError(
            "Gmail service not initialised for this request. "
            "Ensure MailAgent.execute() calls prepare_gmail_service() first."
        )
    return service


# ── Helpers ──────────────────────────────────────────────────────────────


def _parse_message(msg: Dict) -> Dict[str, Any]:
    """Extract useful fields from a Gmail API message resource."""
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    snippet = msg.get("snippet", "")
    labels = msg.get("labelIds", [])
    body = ""
    payload = msg.get("payload", {})
    if payload.get("body", {}).get("data"):
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode(errors="replace")
    elif payload.get("parts"):
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode(errors="replace")
                break

    return {
        "id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "cc": headers.get("Cc", ""),
        "subject": headers.get("Subject", ""),
        "date": headers.get("Date", ""),
        "snippet": snippet,
        "body": body[:5000] if body else snippet,
        "labels": labels,
    }


def _build_mime_message(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    html: bool = False,
) -> str:
    """Create a base64url-encoded RFC 2822 message."""
    mime = MIMEMultipart()
    mime["to"] = to
    mime["from"] = _current_sender_email.get("") or config.gmail_sender_email
    mime["subject"] = subject
    if cc:
        mime["cc"] = cc

    subtype = "html" if html else "plain"
    mime.attach(MIMEText(body, subtype))
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    return raw


# ── SEARCH TOOLS ─────────────────────────────────────────────────────────


@tool("mail_agent")
async def search_messages(
    query: str,
    max_results: int = 10,
    label: str = "",
) -> List[Dict[str, Any]]:
    """
    Search Gmail messages using Gmail search syntax.

    Parameters
    ----------
    query : str
        Gmail search query (same syntax as the Gmail search bar).
        Examples:
          • "from:john@example.com subject:meeting"
          • "has:attachment after:2025/01/01"
          • "is:unread"
          • "mongodb" (keyword search across all mail)
    max_results : int
        Maximum messages to return (1-50).
    label : str
        Gmail label to scope the search (e.g. "INBOX", "SENT").
        Empty string = search all mail (default).

    Returns
    -------
    list of parsed message dicts
    """
    service = _get_service()

    list_kwargs: Dict[str, Any] = {
        "userId": "me",
        "q": query,
        "maxResults": min(max_results, 50),
    }
    if label:
        list_kwargs["labelIds"] = [label]

    results = await asyncio.to_thread(
        service.users()
        .messages()
        .list(**list_kwargs)
        .execute
    )

    messages = results.get("messages", [])
    if not messages:
        return []

    parsed = []
    for meta in messages:
        msg = await asyncio.to_thread(
            service.users()
            .messages()
            .get(userId="me", id=meta["id"], format="full")
            .execute
        )
        parsed.append(_parse_message(msg))

    logger.info("search_messages → query=%s  found=%d", query, len(parsed))
    return parsed


@tool("mail_agent")
async def search_drafts(
    query: str = "",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    List / search Gmail drafts.

    Parameters
    ----------
    query : str
        Optional Gmail search query to filter drafts.
    max_results : int
        Maximum drafts to return.

    Returns
    -------
    list of draft dicts with id, message details
    """
    service = _get_service()
    kwargs: Dict[str, Any] = {"userId": "me", "maxResults": min(max_results, 50)}
    if query:
        kwargs["q"] = query

    results = await asyncio.to_thread(
        service.users().drafts().list(**kwargs).execute
    )
    drafts = results.get("drafts", [])
    if not drafts:
        return []

    parsed = []
    for d in drafts:
        draft = await asyncio.to_thread(
            service.users()
            .drafts()
            .get(userId="me", id=d["id"], format="full")
            .execute
        )
        msg = draft.get("message", {})
        entry = _parse_message(msg)
        entry["draft_id"] = d["id"]
        parsed.append(entry)

    logger.info("search_drafts → query=%s  found=%d", query, len(parsed))
    return parsed


@tool("mail_agent")
async def search_sent_messages(
    query: str = "",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search messages in the SENT folder.

    Parameters
    ----------
    query : str
        Gmail search query.
    max_results : int
        Maximum results.
    """
    return await search_messages(
        query=query,
        max_results=max_results,
        label="SENT",
    )


# ── SEND / COMPOSE TOOLS ────────────────────────────────────────────────


@tool("mail_agent", requires_approval=True)
async def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    html: bool = False,
) -> Dict[str, Any]:
    """
    Send an email through Gmail.

    Parameters
    ----------
    to : str
        Recipient email address(es), comma-separated for multiple.
    subject : str
        Email subject line.
    body : str
        Email body (plain text or HTML).
    cc : str | None
        CC recipients.
    html : bool
        If True, body is treated as HTML.

    Returns
    -------
    dict with status, message_id, thread_id
    """
    service = _get_service()
    raw = _build_mime_message(to, subject, body, cc=cc, html=html)

    sent = await asyncio.to_thread(
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute
    )

    logger.info("send_email → to=%s  message_id=%s", to, sent.get("id"))
    return {
        "status": "sent",
        "message_id": sent.get("id"),
        "thread_id": sent.get("threadId"),
        "to": to,
        "subject": subject,
    }


@tool("mail_agent")
async def draft_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    html: bool = False,
) -> Dict[str, Any]:
    """
    Create a draft in Gmail (not sent).

    Returns
    -------
    dict with status, draft_id, message details
    """
    service = _get_service()
    raw = _build_mime_message(to, subject, body, cc=cc, html=html)

    draft = await asyncio.to_thread(
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw}})
        .execute
    )

    logger.info("draft_email → to=%s  draft_id=%s", to, draft.get("id"))
    return {
        "status": "drafted",
        "draft_id": draft.get("id"),
        "message_id": draft.get("message", {}).get("id"),
        "to": to,
        "subject": subject,
    }


@tool("mail_agent")
async def get_message_by_id(
    message_id: str,
) -> Dict[str, Any]:
    """
    Fetch a single Gmail message by its ID.

    Useful when you already have a message ID from a search result
    and need the full body.
    """
    service = _get_service()
    msg = await asyncio.to_thread(
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute
    )
    return _parse_message(msg)


@tool("mail_agent", requires_approval=True)
async def reply_to_message(
    message_id: str,
    body: str,
    html: bool = False,
) -> Dict[str, Any]:
    """
    Reply to an existing Gmail message (preserves thread).

    Parameters
    ----------
    message_id : str
        The ID of the message to reply to.
    body : str
        Reply body.
    """
    service = _get_service()
    original = await asyncio.to_thread(
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="metadata", metadataHeaders=["From", "Subject", "Message-ID"])
        .execute
    )
    headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}
    thread_id = original.get("threadId")

    reply_to = headers.get("From", "")
    subject = headers.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    raw = _build_mime_message(reply_to, subject, body, html=html)

    sent = await asyncio.to_thread(
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw, "threadId": thread_id})
        .execute
    )

    logger.info("reply_to_message → original=%s  new=%s", message_id, sent.get("id"))
    return {
        "status": "sent",
        "message_id": sent.get("id"),
        "thread_id": sent.get("threadId"),
        "in_reply_to": message_id,
        "to": reply_to,
        "subject": subject,
    }
