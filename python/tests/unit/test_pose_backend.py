"""Unit tests for pose_backend torso angle helper (no YOLO model)."""
from __future__ import annotations

import math

import numpy as np

from people_analytics_service.pose_backend import _torso_angle_from_vertical


def _kpts(upright: bool) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[5] = (40, 100, 0.9)   # left shoulder
    k[6] = (80, 100, 0.9)   # right shoulder
    if upright:
        k[11] = (45, 180, 0.9)  # left hip
        k[12] = (75, 180, 0.9)  # right hip
    else:
        k[11] = (120, 105, 0.9)  # horizontal torso
        k[12] = (160, 108, 0.9)
    return k


def test_upright_torso_near_zero():
    angle = _torso_angle_from_vertical(_kpts(upright=True), 0.5)
    assert angle is not None
    assert angle < 15.0


def test_horizontal_torso_large_angle():
    angle = _torso_angle_from_vertical(_kpts(upright=False), 0.5)
    assert angle is not None
    assert angle > 60.0
