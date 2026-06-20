"""Alembic migration environment.

Uses psycopg2 (sync) for running migrations — asyncpg is only for the
live service. Both read DATABASE_URL from the environment; no credentials
are stored in alembic.ini.

Run from the python/ directory:
    alembic upgrade head
    alembic downgrade -1
    alembic revision --autogenerate -m "describe change"
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the people_analytics_service package importable without installing it
sys.path.insert(0, str(Path(__file__).parent.parent))

from people_analytics_service.db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_url(url: str) -> str:
    """Strip asyncpg driver specifier so psycopg2 can be used for migrations."""
    return (
        url
        .replace("postgresql+asyncpg://", "postgresql://", 1)
        .replace("postgres+asyncpg://",   "postgresql://", 1)
        .replace("postgres://",            "postgresql://", 1)
    )


def _get_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise ValueError(
            "DATABASE_URL is not set. "
            "Export it before running Alembic:\n"
            "  export DATABASE_URL=postgresql://user:pass@host:port/db"
        )
    return _sync_url(url)


def run_migrations_offline() -> None:
    """Generate SQL script without a live DB connection (--sql mode)."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    config.set_main_option("sqlalchemy.url", _get_url())
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
