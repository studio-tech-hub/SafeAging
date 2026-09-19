"""Ordered list of pipeline stages run by _run_person_detection_pipeline.

See pipeline/__init__.py for why DecodeStage isn't one of these.
"""
from .base import Stage
from .preprocess import PreprocessStage
from .detect import DetectStage
from .track import TrackStage
from .fall import FallStage
from .face import FaceStage
from .zone import ZoneStage
from .persist import PersistStage

PIPELINE_STAGES: list[Stage] = [
    PreprocessStage(),
    DetectStage(),
    TrackStage(),
    FallStage(),
    FaceStage(),
    ZoneStage(),
    PersistStage(),
]

__all__ = [
    "Stage",
    "PreprocessStage",
    "DetectStage",
    "TrackStage",
    "FallStage",
    "FaceStage",
    "ZoneStage",
    "PersistStage",
    "PIPELINE_STAGES",
]
