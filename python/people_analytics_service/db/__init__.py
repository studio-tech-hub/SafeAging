"""Database layer for the SafeAging analytics service.

Usage (P2-2 — preferred for new code):
    from people_analytics_service.db import get_session, get_repository

    async with get_session() as session:
        repo = get_repository()
        persons = await repo.list_persons(session)

`dal` (below) is still exported for existing call sites and for db.repository
itself, which is a thin, named pass-through over it — see
db/repository.py's module docstring for why call sites should prefer
get_repository()/Repository over importing dal directly going forward.
"""
from .session import dispose_engine, get_session, init_engine, is_edge_mode, is_ready
from . import dal
from .repository import Repository, get_repository

__all__ = [
    "get_session",
    "init_engine",
    "dispose_engine",
    "is_edge_mode",
    "is_ready",
    "dal",
    "get_repository",
    "Repository",
]
