"""Narrow Stage interface every pipeline stage implements (P2-1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import InferContext


class Stage(ABC):
    """One named, independently-testable step of the /infer pipeline.

    Contract: mutate `ctx` in place; return nothing. Set `ctx.stopped = True`
    and `ctx.result = <value>` to short-circuit the remaining stages (the
    orchestrator in api.py checks `ctx.stopped` after every stage) — this is
    the only stage in the current pipeline that needs this is DetectStage,
    mirroring the original function's early `return []` on a YOLO error.
    """

    @abstractmethod
    def run(self, ctx: "InferContext") -> None:
        raise NotImplementedError
