"""
Multi-Agentic RAG System — application entry point.
"""

from __future__ import annotations

import logging
import sys

import pathlib
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import register_middleware
from api.routes import router as api_router
from api.streaming import router as stream_router
from auth.routes import router as auth_router
from connectors.routes import router as connector_router
from config.settings import config
from core.agent_factory import build_agent_instances
from connectors.registry import ConnectorRegistry
from database.helpers import ensure_demo_user, expire_stale_hitl_requests
from database.session import async_session_factory
from tools.registry import ToolRegistry
from utils.validators import validate_tools_for_agents

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
for _noisy in ("httpcore", "httpx", "openai", "urllib3", "hpack"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

logging.getLogger("botocore").setLevel(logging.WARNING)
def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Discovering tools…")
        registry = ToolRegistry()
        registry.auto_discover_tools()

        logger.info("Validating tool ↔ agent bindings…")
        agents = build_agent_instances(registry)
        validate_tools_for_agents(agents, registry)

        # Discover OAuth connectors
        connector_registry = ConnectorRegistry()
        connector_registry.discover()

        # Clean up orphaned HITL requests from previous server instances
        expired = await expire_stale_hitl_requests()
        if expired:
            logger.info("Cleaned up %d stale HITL requests from previous run", expired)

        hitl_tools = registry.get_hitl_tools_for_agent_task(registry.list_tools())
        if hitl_tools:
            logger.info("HITL-protected tools: %s", hitl_tools)

        # Seed optional demo login for public testing.
        if config.demo_user_enabled:
            try:
                async with async_session_factory() as session:
                    demo = await ensure_demo_user(session)
                    await session.commit()
                if demo:
                    logger.info("Demo user ready: %s", demo["email"])
            except Exception:
                logger.exception("Failed to seed demo user")

        logger.info("Application ready to accept requests.")
        yield

    app = FastAPI(
        title="Multi-Agentic RAG System",
        version="3.0.0",
        description="Production multi-agent RAG with Connectors, HITL",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_middleware(app)

    # Routes
    app.include_router(auth_router, prefix="/api/v1/auth")
    app.include_router(connector_router, prefix="/api/v1/connectors")
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(stream_router, prefix="/api/v1")

    # ── Public shared conversation routes (no auth) ─────────────────
    @app.get("/api/v1/shared/{share_token}")
    async def get_shared_conversation(share_token: str):
        from sqlalchemy.ext.asyncio import AsyncSession
        from database.helpers import load_shared_conversation
        from database.session import async_session_factory

        async with async_session_factory() as session:
            data = await load_shared_conversation(session, share_token)
        if data is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Shared conversation not found")
        return data

    @app.get("/shared/{share_token}", response_class=HTMLResponse)
    async def shared_page(share_token: str):
        page = pathlib.Path(__file__).resolve().parent / "frontend" / "shared.html"
        if not page.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Shared page not found")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    frontend_dir = pathlib.Path(__file__).resolve().parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()

if __name__ == "__main__":
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        reload_excludes=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"] if config.debug else None,
        log_level="debug" if config.debug else "info",
    )
