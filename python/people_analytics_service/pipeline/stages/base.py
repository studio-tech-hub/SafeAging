"""Narrow Stage interface every pipeline stage implements (P2-1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import AppConfig
    from ..context import InferContext


class Stage(ABC):
    """One named, independently-testable step of the /infer pipeline.

    Contract: mutate `ctx` in place; return nothing. Set `ctx.stopped = True`
    and `ctx.result = <value>` to short-circuit the remaining stages (the
    orchestrator in api.py checks `ctx.stopped` after every stage) — this is
    the only stage in the current pipeline that needs this is DetectStage,
    mirroring the original function's early `return []` on a YOLO error.

    P2-3: `config` is passed explicitly by the orchestrator (one
    `get_config()` call per request, not per stage) rather than each stage
    importing config.py's module-level globals directly. Stage instances
    themselves stay cheap, reusable singletons (see
    pipeline/stages/__init__.py's PIPELINE_STAGES) constructed once at import
    time with no config dependency of their own — only `run()` is
    config-aware, which keeps a future hot-reloadable AppConfig (re-fetched
    per request via get_config()) a pure drop-in with no stage-construction
    changes needed.
    """

    @abstractmethod
    def run(self, ctx: "InferContext", config: "AppConfig") -> None:
        raise NotImplementedError
