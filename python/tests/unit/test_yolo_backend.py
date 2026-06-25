"""Unit tests for YOLO backend name resolution."""

import pytest


def test_resolve_backend_cpu(monkeypatch):
    monkeypatch.setenv("YOLO_BACKEND", "cpu")
    from people_analytics_service import config
    from people_analytics_service import yolo_backend

    config.CONFIG.__init__()
    assert yolo_backend._resolve_backend_name() == "cpu"


def test_resolve_backend_qnn_alias(monkeypatch):
    monkeypatch.setenv("YOLO_BACKEND", "qnn")
    from people_analytics_service import config
    from people_analytics_service import yolo_backend

    config.CONFIG.__init__()
    assert yolo_backend._resolve_backend_name() == "qnn_htp"


def test_resolve_backend_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("YOLO_BACKEND", "cuda")
    from people_analytics_service import config
    from people_analytics_service import yolo_backend

    config.CONFIG.__init__()
    assert yolo_backend._resolve_backend_name() == "cpu"


def test_qnn_runtime_status_has_providers():
    from people_analytics_service import yolo_backend

    status = yolo_backend.qnn_runtime_status()
    assert "available_providers" in status
    assert "qnn_execution_provider" in status
    assert status["note"] == "Qualcomm QNN plugin — not NVIDIA CUDA"


def test_yolo_imgsz_clamped_to_minimum_640(monkeypatch):
    monkeypatch.setenv("YOLO_IMGSZ", "256")
    monkeypatch.setenv("FACE_DET_SIZE", "320")
    from people_analytics_service import config

    config.CONFIG.__init__()
    assert config.CONFIG.yolo_imgsz >= 640
    assert config.CONFIG.face_det_size >= 640
    assert config.MIN_INPUT_RESOLUTION >= 640
