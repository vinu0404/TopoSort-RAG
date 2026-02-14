"""
Gmail agent tools — full integration with Gmail API via Google OAuth2.

Supports two token modes:
  1. Per-user OAuth tokens (via connectors module) — preferred
  2. Legacy file-based tokens  (via .env config)    — fallback
"""

from __future__ import annotations

import base64
import contextvars
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from config.settings import config
from tools import tool

logger = logging.getLogger(__name__)

# ── Context variable for per-user token injection ────────────────────────
# The mail agent sets this before calling tool functions so that
# _get_gmail_service() can load the correct user's OAuth token.
current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default=""
)

_gmail_service = None            # legacy file-based singleton
_user_services: Dict[str, Any] = {}  # per-user service cache

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _build_service_from_token(access_token: str):
    """Build a Gmail service from a raw OAuth access token (per-user)."""
    creds = Credentials(token=access_token)
    return build("gmail", "v1", credentials=creds)


def _get_gmail_service():
    """
    Build & return a Gmail API service object.

    Priority:
      1. Per-user token via connectors (if current_user_id is set)
      2. Legacy file-based token (if GMAIL_CREDENTIALS_FILE is configured)
    """
    user_id = current_user_id.get("")

    # ── Per-user connector token ─────────────────────────────────────
    if user_id:
        try:
            import asyncio
            from connectors.token_manager import get_active_token

            # We're inside an async context, use the running loop
            loop = asyncio.get_running_loop()
            # Create a future and run in a helper task
            token = None

            async def _fetch():
                return await get_active_token(user_id, "gmail")

            if user_id in _user_services:
                return _user_services[user_id]

        except RuntimeError:
            pass  # No running loop — fall through to legacy

    # ── Legacy file-based token ──────────────────────────────────────
    global _gmail_service
    if _gmail_service is not None:
        return _gmail_service

    creds_file = config.gmail_credentials_file
    token_file = config.gmail_token_file

    creds = None
    if token_file and os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_file or not os.path.exists(creds_file):
                raise RuntimeError(
                    "Gmail not connected. Either connect Gmail via the UI "
                    "(Settings → Connect Gmail) or set GMAIL_CREDENTIALS_FILE "
                    "in .env for legacy file-based auth."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)
        if token_file:
            os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
            with open(token_file, "w") as f:
                f.write(creds.to_json())

    _gmail_service = build("gmail", "v1", credentials=creds)
    return _gmail_service


async def get_gmail_service_async(user_id: str = ""):
    """
    Async version — tries per-user connector token first, then legacy.
    Call this from the mail agent instead of _get_gmail_service().
    """
    if user_id:
        from connectors.token_manager import get_active_token

        token = await get_active_token(user_id, "gmail")
        if token:
            # Build and cache service
            service = _build_service_from_token(token)
            _user_services[user_id] = service
            return service

    # Fallback to legacy sync approach
    return _get_gmail_service()


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
    mime["from"] = config.gmail_sender_email
    mime["subject"] = subject
    if cc:
        mime["cc"] = cc

    subtype = "html" if html else "plain"
    mime.attach(MIMEText(body, subtype))
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    return raw


# ── SEARCH TOOLS ─────

@tool("mail_agent")
async def search_messages(
    query: str,
    max_results: int = 10,
    label: str = "INBOX",
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
    max_results : int
        Maximum messages to return (1-50).
    label : str
        Gmail label to scope the search (default INBOX).

    Returns
    -------
    list of parsed message dicts
    """
    service = _get_gmail_service()
    results = (
        service.users()
        .messages()
        .list(userId="me", q=query, labelIds=[label], maxResults=min(max_results, 50))
        .execute()
    )

    messages = results.get("messages", [])
    if not messages:
        return []

    parsed = []
    for meta in messages:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=meta["id"], format="full")
            .execute()
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
    service = _get_gmail_service()
    kwargs: Dict[str, Any] = {"userId": "me", "maxResults": min(max_results, 50)}
    if query:
        kwargs["q"] = query

    results = service.users().drafts().list(**kwargs).execute()
    drafts = results.get("drafts", [])
    if not drafts:
        return []

    parsed = []
    for d in drafts:
        draft = (
            service.users()
            .drafts()
            .get(userId="me", id=d["id"], format="full")
            .execute()
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


# ── SEND / COMPOSE TOOLS ───────

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
    service = _get_gmail_service()
    raw = _build_mime_message(to, subject, body, cc=cc, html=html)

    sent = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute()
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
    service = _get_gmail_service()
    raw = _build_mime_message(to, subject, body, cc=cc, html=html)

    draft = (
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw}})
        .execute()
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
    service = _get_gmail_service()
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
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
    service = _get_gmail_service()
    original = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="metadata", metadataHeaders=["From", "Subject", "Message-ID"])
        .execute()
    )
    headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}
    thread_id = original.get("threadId")

    reply_to = headers.get("From", "")
    subject = headers.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    raw = _build_mime_message(reply_to, subject, body, html=html)

    sent = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw, "threadId": thread_id})
        .execute()
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
