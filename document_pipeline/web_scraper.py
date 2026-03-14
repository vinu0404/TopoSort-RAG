"""
Web scraper — crawls URLs with configurable depth and processes content
into chunks suitable for the RAG pipeline.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from document_pipeline.chunker import StructureAwareChunker
from document_pipeline.embedder import get_embedding_model
from document_pipeline.vector_store import get_vector_store

logger = logging.getLogger(__name__)

MAX_PAGES_PER_URL = 20
REQUEST_TIMEOUT = 15
USER_AGENT = "MRAG-WebScraper/1.0"

_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}


async def scrape_url(url: str, depth: int = 1) -> List[Dict[str, str]]:
    """Scrape a URL and optionally follow same-domain links up to *depth* levels.

    depth=1 means just the given page.
    depth=2 means the given page + pages linked from it, etc.

    Returns list of ``{"url": str, "title": str, "content": str}``.
    """
    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc

    visited: set[str] = set()
    pages: List[Dict[str, str]] = []
    queue: deque[tuple[str, int]] = deque([(url, 0)])

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        while queue and len(pages) < MAX_PAGES_PER_URL:
            current_url, current_depth = queue.popleft()

            # Normalise
            normalised = current_url.rstrip("/")
            if normalised in visited:
                continue
            visited.add(normalised)

            try:
                resp = await client.get(current_url)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", current_url, e)
                continue

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            # Remove unwanted elements
            for tag in soup.find_all(_STRIP_TAGS):
                tag.decompose()

            # Extract text
            body = soup.find("body")
            text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

            if text.strip():
                pages.append({"url": current_url, "title": title, "content": text})

            # Follow links if within depth
            if current_depth + 1 < depth:
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    full_url = urljoin(current_url, href)
                    parsed = urlparse(full_url)

                    # Same domain only, no fragments, no non-http
                    if parsed.netloc != base_domain:
                        continue
                    if parsed.scheme not in ("http", "https"):
                        continue

                    clean = parsed._replace(fragment="").geturl()
                    if clean.rstrip("/") not in visited:
                        queue.append((clean, current_depth + 1))

    logger.info("Scraped %d pages from %s (depth=%d)", len(pages), url, depth)
    return pages


async def process_scraped_pages(
    user_id: str,
    web_collection_id: str,
    pages: List[Dict[str, str]],
) -> Dict[str, int]:
    """Chunk, embed, and store scraped pages into the user's Qdrant collection.

    Returns ``{"pages_processed": int, "chunks_created": int}``.
    """
    store = get_vector_store()
    embed = get_embedding_model()
    chunker = StructureAwareChunker(chunk_size=1024)

    await store.create_user_collection(user_id)

    total_chunks = 0

    for page in pages:
        doc = {
            "content": page["content"],
            "metadata": {
                "filename": page.get("title") or page["url"],
                "doc_type": "web_page",
                "uploaded_at": "",
            },
        }

        chunks = await chunker.chunk_document(doc)
        if not chunks:
            continue

        n = await store.add_web_page(
            user_id=user_id,
            web_collection_id=web_collection_id,
            page=page,
            chunks=chunks,
            embed_fn=embed.embed,
            embed_batch_fn=embed.embed_batch,
        )
        total_chunks += n

    return {"pages_processed": len(pages), "chunks_created": total_chunks}
