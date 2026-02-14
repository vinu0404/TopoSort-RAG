"""
Multi-Agentic RAG System — application entry point.
"""

from __future__ import annotations

import logging
import sys

import pathlib

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.middleware import register_middleware
from api.routes import router as api_router
from api.streaming import router as stream_router
from api.auth import router as auth_router
from config.settings import config
from core.agent_factory import build_agent_instances
from database.helpers import expire_stale_hitl_requests
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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Agentic RAG System",
        version="1.0.0",
        description="Production multi-agent RAG with streaming.",
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
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(stream_router, prefix="/api/v1")

    frontend_dir = pathlib.Path(__file__).resolve().parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    @app.on_event("startup")
    async def on_startup():
        logger.info("Discovering tools…")
        registry = ToolRegistry()
        registry.auto_discover_tools()

        logger.info("Validating tool ↔ agent bindings…")
        agents = build_agent_instances(registry)
        validate_tools_for_agents(agents, registry)

        # Clean up orphaned HITL requests from previous server instances
        expired = await expire_stale_hitl_requests()
        if expired:
            logger.info("Cleaned up %d stale HITL requests from previous run", expired)

        hitl_tools = registry.get_hitl_tools_for_agent_task(registry.list_tools())
        if hitl_tools:
            logger.info("HITL-protected tools: %s", hitl_tools)

        logger.info("Application ready to accept requests.")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info",
    )
