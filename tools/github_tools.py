"""
GitHub tools — repo info, issues, PRs, repo creation, PR creation.

Architecture:
  • GitHubAgent.execute() calls ``get_active_token(user_id, "github")``
    once per request to get a fresh OAuth token.
  • The token is passed directly to each tool function (token pass-through
    pattern — no ContextVar needed since the SDK is just httpx REST calls).
  • All calls are natively async via httpx — no ``to_thread`` needed.

HITL:
  • Read-only tools: ``get_repo_info``, ``list_repo_issues``, ``list_pull_requests``
    → no approval required.
  • Mutating tools: ``create_repo``, ``create_pull_request``
    → ``requires_approval=True`` → HITL dialog fires automatically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from tools import tool

logger = logging.getLogger(__name__)

_GH_API = "https://api.github.com"


def _log_gh_error(resp: httpx.Response, tool_name: str) -> None:
    """Log GitHub API error details including required permissions."""
    accepted = resp.headers.get("X-Accepted-GitHub-Permissions", "(not set)")
    oauth_scopes = resp.headers.get("X-OAuth-Scopes", "(not set)")
    try:
        body = resp.json()
    except Exception:
        body = resp.text[:500]
    logger.error(
        "[%s] GitHub %d — body=%s | X-Accepted-GitHub-Permissions=%s | X-OAuth-Scopes=%s",
        tool_name, resp.status_code, body, accepted, oauth_scopes,
    )


def _gh_headers(token: str) -> Dict[str, str]:
    """Standard GitHub API headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ── Read-only tools (no approval) ─────────────────────────────────────────


@tool("github_agent")
async def list_user_repos(
    token: str,
    sort: str = "updated",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    List repositories for the authenticated user.

    Parameters
    ----------
    sort : str
        Sort by: created, updated, pushed, full_name (default: updated).
    limit : int
        Maximum repos to return (1-100).
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/user/repos",
            headers=_gh_headers(token),
            params={
                "sort": sort,
                "per_page": min(limit, 100),
                "affiliation": "owner,collaborator,organization_member",
            },
        )
        resp.raise_for_status()

    return [
        {
            "full_name": r["full_name"],
            "description": r.get("description"),
            "language": r.get("language"),
            "stars": r["stargazers_count"],
            "private": r["private"],
            "html_url": r["html_url"],
            "updated_at": r.get("updated_at"),
        }
        for r in resp.json()
    ]


@tool("github_agent")
async def get_repo_info(token: str, owner: str, repo: str) -> Dict[str, Any]:
    """Get repository metadata — description, stars, language, visibility."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/repos/{owner}/{repo}",
            headers=_gh_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

    return {
        "full_name": data["full_name"],
        "description": data.get("description"),
        "language": data.get("language"),
        "stars": data["stargazers_count"],
        "forks": data["forks_count"],
        "open_issues": data["open_issues_count"],
        "visibility": data.get("visibility", "public"),
        "default_branch": data["default_branch"],
        "html_url": data["html_url"],
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


@tool("github_agent")
async def list_repo_issues(
    token: str,
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """List issues in a repository (excludes pull requests)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/repos/{owner}/{repo}/issues",
            headers=_gh_headers(token),
            params={"state": state, "per_page": min(limit, 30)},
        )
        resp.raise_for_status()

    return [
        {
            "number": i["number"],
            "title": i["title"],
            "state": i["state"],
            "user": i["user"]["login"],
            "labels": [lbl["name"] for lbl in i.get("labels", [])],
            "created_at": i.get("created_at"),
            "html_url": i["html_url"],
        }
        for i in resp.json()
        if "pull_request" not in i  # GitHub mixes issues + PRs in this endpoint
    ]


@tool("github_agent")
async def list_pull_requests(
    token: str,
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """List pull requests in a repository."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GH_API}/repos/{owner}/{repo}/pulls",
            headers=_gh_headers(token),
            params={"state": state, "per_page": min(limit, 30)},
        )
        resp.raise_for_status()

    return [
        {
            "number": pr["number"],
            "title": pr["title"],
            "state": pr["state"],
            "user": pr["user"]["login"],
            "head": pr["head"]["ref"],
            "base": pr["base"]["ref"],
            "draft": pr.get("draft", False),
            "created_at": pr.get("created_at"),
            "html_url": pr["html_url"],
        }
        for pr in resp.json()
    ]


# ── Mutating tools (HITL required) ──────────────────────────────────────


@tool("github_agent", requires_approval=True)
async def create_repo(
    token: str,
    name: str,
    description: str = "",
    private: bool = False,
) -> Dict[str, Any]:
    """
    Create a new GitHub repository on the authenticated user's account.
    ⚠️ requires_approval — creates a real repository.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_GH_API}/user/repos",
            headers=_gh_headers(token),
            json={
                "name": name,
                "description": description,
                "private": private,
                "auto_init": True,
            },
        )
        if resp.status_code >= 400:
            _log_gh_error(resp, "create_repo")
        resp.raise_for_status()
        data = resp.json()

    logger.info("create_repo → %s (private=%s)", data["full_name"], data["private"])
    return {
        "full_name": data["full_name"],
        "html_url": data["html_url"],
        "private": data["private"],
        "default_branch": data["default_branch"],
    }


@tool("github_agent", requires_approval=True)
async def create_pull_request(
    token: str,
    owner: str,
    repo: str,
    title: str,
    head: str,
    base: str,
    body: str = "",
    draft: bool = False,
) -> Dict[str, Any]:
    """
    Create a pull request on a repository.
    ⚠️ requires_approval — opens a real PR visible to collaborators.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_GH_API}/repos/{owner}/{repo}/pulls",
            headers=_gh_headers(token),
            json={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
                "draft": draft,
            },
        )
        if resp.status_code >= 400:
            _log_gh_error(resp, "create_pull_request")
        resp.raise_for_status()
        data = resp.json()

    logger.info("create_pull_request → %s/%s#%d", owner, repo, data["number"])
    return {
        "number": data["number"],
        "title": data["title"],
        "html_url": data["html_url"],
        "state": data["state"],
        "head": head,
        "base": base,
        "draft": data.get("draft", False),
    }
