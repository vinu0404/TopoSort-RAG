"""
Web search agent tools — powered by Tavily Search API.

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
import aiohttp

from config.settings import config
from tools import tool

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

@tool("web_search_agent", requires_approval=True)
async def web_search(
    query: str,
    num_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
    topic: str = "general",
) -> Dict[str, Any]:
    """
    Search the web via the Tavily Search API.

    Parameters
    ----------
    query : str
        The search query.
    num_results : int
        Max number of results to return (1-20).
    search_depth : str
        "basic" (fast) or "advanced" (deeper extraction, more tokens).
    include_answer : bool
        If True, Tavily returns a short LLM-generated answer alongside results.
    include_raw_content : bool
        If True, each result includes raw page content (increases response size).
    topic : str
        "general" or "news" — news restricts to recent articles.

    Returns
    -------
    dict with keys: answer, results (list), query, response_time
    """
    api_key = config.tavily_api_key
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file."
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "max_results": min(num_results, 20),
        "topic": topic,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(TAVILY_SEARCH_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": r.get("score", 0.0),
            "raw_content": r.get("raw_content"),
        })

    logger.info(
        "tavily_search → query=%s  results=%d  depth=%s",
        query, len(results), search_depth,
    )

    return {
        "answer": data.get("answer"),
        "results": results,
        "query": data.get("query", query),
        "response_time": data.get("response_time"),
    }


# ── Tavily extract (fetch + parse a URL) ──────────────────────────────────

@tool("web_search_agent")
async def fetch_url(url: str) -> Dict[str, Any]:
    """
    Extract clean content from a URL using the Tavily Extract API.

    Falls back to raw HTTP fetch if Tavily extract is unavailable.
    """
    api_key = config.tavily_api_key
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    TAVILY_EXTRACT_URL,
                    json={"api_key": api_key, "urls": [url]},
                )
                resp.raise_for_status()
                data = resp.json()

            extracted = data.get("results", [])
            if extracted:
                page = extracted[0]
                return {
                    "url": url,
                    "status": 200,
                    "content": page.get("raw_content", "")[:15_000],
                    "title": page.get("title", ""),
                    "extraction_method": "tavily_extract",
                }
        except Exception as exc:
            logger.warning("Tavily extract failed for %s: %s — falling back", url, exc)


    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                text = await resp.text()
                return {
                    "url": url,
                    "status": resp.status,
                    "content": text[:15_000],
                    "extraction_method": "raw_http",
                }
    except Exception as exc:
        return {"url": url, "error": str(exc)}


# ── Tavily topic-specific helpers ─────────────────────────────────────────

@tool("web_search_agent", requires_approval=True)
async def web_search_news(
    query: str,
    num_results: int = 5,
) -> Dict[str, Any]:
    """Search recent news articles via Tavily (topic='news')."""
    return await web_search(
        query=query,
        num_results=num_results,
        search_depth="basic",
        topic="news",
    )


@tool("web_search_agent", requires_approval=True)
async def web_search_deep(
    query: str,
    num_results: int = 5,
) -> Dict[str, Any]:
    """Deep web search via Tavily (search_depth='advanced', includes raw content)."""
    return await web_search(
        query=query,
        num_results=num_results,
        search_depth="advanced",
        include_raw_content=True,
    )
