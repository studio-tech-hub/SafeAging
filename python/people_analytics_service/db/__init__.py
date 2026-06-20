"""Database layer for the SafeAging analytics service.

Usage:
    from people_analytics_service.db import get_session, init_engine
    from people_analytics_service.db import dal

    async with get_session() as session:
        persons = await dal.list_persons(session)
"""
from .session import dispose_engine, get_session, init_engine, is_edge_mode, is_ready
from . import dal

__all__ = [
    "get_session",
    "init_engine",
    "dispose_engine",
    "is_edge_mode",
    "is_ready",
    "dal",
]
