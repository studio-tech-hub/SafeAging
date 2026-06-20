"""Async SQLAlchemy engine and session factory.

When DATABASE_URL is empty, falls back to EDGE_SQLITE_PATH (edge SQLite store).

Lifecycle
---------
1. call init_engine(url) once at service startup
2. use get_session() as an async context manager per request / task
3. call await dispose_engine() at service shutdown
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine = None
_session_factory: async_sessionmaker | None = None
_edge_mode = False


class _EdgeSession:
    """Placeholder session when using edge SQLite (operations auto-commit)."""

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


_EDGE_SESSION = _EdgeSession()

_POOL_SIZE = 5
_POOL_OVERFLOW = 10
_POOL_RECYCLE_SEC = 1800  # 30 min — avoids stale connections on AI Box


def is_edge_mode() -> bool:
    return _edge_mode


def _to_async_url(url: str) -> str:
    """Convert a plain postgresql:// URL to the asyncpg dialect form."""
    for plain, async_ in (
        ("postgresql://", "postgresql+asyncpg://"),
        ("postgres://",   "postgresql+asyncpg://"),
    ):
        if url.startswith(plain):
            return url.replace(plain, async_, 1)
    return url  # already has a driver or is empty


def init_engine(database_url: str | None = None) -> None:
    """Create Postgres engine or open edge SQLite when DATABASE_URL is empty."""
    global _engine, _session_factory, _edge_mode

    url = (database_url or os.getenv("DATABASE_URL", "")).strip()
    if not url:
        from .edge_store import init_edge_store

        edge_path = os.getenv(
            "EDGE_SQLITE_PATH", "/app/runtime/edge_state.db"
        ).strip()
        if not edge_path:
            raise ValueError(
                "DATABASE_URL is not configured and EDGE_SQLITE_PATH is empty."
            )
        init_edge_store(edge_path)
        _engine = None
        _session_factory = None
        _edge_mode = True
        return

    _edge_mode = False
    _engine = create_async_engine(
        _to_async_url(url),
        pool_size=_POOL_SIZE,
        max_overflow=_POOL_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=_POOL_RECYCLE_SEC,
        echo=False,
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def dispose_engine() -> None:
    """Drain Postgres pool or close edge SQLite handle."""
    global _engine, _edge_mode
    if _engine is not None:
        await _engine.dispose()
        _engine = None
    if _edge_mode:
        from .edge_store import close_edge_store

        close_edge_store()
        _edge_mode = False


def is_ready() -> bool:
    """Return True if Postgres or edge SQLite is available."""
    if _engine is not None:
        return True
    if _edge_mode:
        from .edge_store import is_initialized

        return is_initialized()
    return False


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession | object, None]:
    """Provide a transactional async session (Postgres) or edge placeholder."""
    if _edge_mode:
        yield _EDGE_SESSION
        return

    if _session_factory is None:
        raise RuntimeError(
            "Database session factory is not ready. "
            "Ensure init_engine() was called and succeeded."
        )

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
