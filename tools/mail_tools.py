"""
Gmail agent tools — full integration with Gmail API via Google OAuth2.

"""

from __future__ import annotations

import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from config.settings import config
from tools import tool

logger = logging.getLogger(__name__)

_gmail_service = None

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

def _get_gmail_service():
    """Build & cache a Gmail API service object."""
    global _gmail_service
    if _gmail_service is not None:
        return _gmail_service


    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/gmail.modify",
    ]

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
                    "Gmail credentials not configured. Set GMAIL_CREDENTIALS_FILE "
                    "in .env to your OAuth2 credentials.json path and run the app "
                    "once locally to complete the consent screen."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)
        if token_file:
            os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
            with open(token_file, "w") as f:
                f.write(creds.to_json())

    _gmail_service = build("gmail", "v1", credentials=creds)
    return _gmail_service


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
