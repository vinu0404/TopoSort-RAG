from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Pull DB URL from the app's own settings so there is one source of truth.
# DATABASE_URL must use the asyncpg driver (postgresql+asyncpg://...).
from config.settings import config as app_config

alembic_config = context.config
alembic_config.set_main_option("sqlalchemy.url", app_config.database_url)

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# Import every model module here so autogenerate sees the full schema.
from database.models import Base  # noqa: F401 — registers all ORM models

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection (alembic upgrade --sql)."""
    url = alembic_config.get_main_option("sqlalchemy.url")
    # asyncpg is async-only — strip the driver suffix for offline SQL generation
    offline_url = url.replace("+asyncpg", "")
    context.configure(
        url=offline_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,       # detect column type changes in autogenerate
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # never reuse connections across migration runs
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
