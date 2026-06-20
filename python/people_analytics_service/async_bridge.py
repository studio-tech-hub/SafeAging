"""Run async DB coroutines from sync /infer worker threads safely.

Background cache refreshers (zone_engine, face_engine, config_engine) must NOT
create their own asyncio event loops — that leaks SQLAlchemy pool connections
and eventually exhausts PostgreSQL max_connections (UI admin APIs return 500).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_main_loop: asyncio.AbstractEventLoop | None = None
_init_lock = threading.Lock()


def init_async_bridge(loop: asyncio.AbstractEventLoop) -> None:
    """Call once from FastAPI startup after the event loop is running."""
    global _main_loop
    with _init_lock:
        _main_loop = loop


def run_async(coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
    """Block the calling thread until *coro* completes on the main loop."""
    loop = _main_loop
    if loop is None or not loop.is_running():
        raise RuntimeError("async_bridge is not initialised (service still starting?)")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)
