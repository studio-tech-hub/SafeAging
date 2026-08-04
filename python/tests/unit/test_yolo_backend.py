"""Unit tests for YOLO backend name resolution and inference postprocessing."""

import numpy as np
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


# ---------------------------------------------------------------------------
# _nms_indices (P1-3: xyxy-native greedy NMS via tracking.greedy_nms_indices,
# replaces cv2.dnn.NMSBoxes which silently expected [x, y, w, h] input)
# ---------------------------------------------------------------------------

class TestNmsIndices:
    def test_empty_input(self):
        from people_analytics_service.yolo_backend import _nms_indices

        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        assert _nms_indices(boxes, scores, 0.5) == []

    def test_duplicate_suppressed(self):
        from people_analytics_service.yolo_backend import _nms_indices

        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        assert _nms_indices(boxes, scores, 0.5) == [0]

    def test_translation_invariant_regression(self):
        """P1-3: this exact box pair previously flipped NMS outcome (kept near
        the frame origin, wrongly suppressed far from it) when xyxy boxes were
        fed into cv2.dnn.NMSBoxes as if they were [x, y, w, h]. True IoU here
        is ~0.39, below the 0.45 threshold, so both boxes must survive in
        both cases -- NMS must not depend on absolute position in the frame.
        """
        from people_analytics_service.yolo_backend import _nms_indices

        near = np.array([[0.0, 0.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        far = np.array([[300.0, 300.0, 340.0, 340.0], [310.0, 310.0, 350.0, 350.0]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        assert _nms_indices(near, scores, 0.45) == [0, 1]
        assert _nms_indices(far, scores, 0.45) == [0, 1]


# ---------------------------------------------------------------------------
# _row_is_canvas_space (P1-3: end2end letterbox coordinate-space detection)
# ---------------------------------------------------------------------------

class TestRowIsCanvasSpace:
    def test_forced_canvas(self):
        from people_analytics_service.yolo_backend import _row_is_canvas_space

        # Even coordinates that look "already in frame" must be treated as
        # canvas-space once explicitly forced.
        assert _row_is_canvas_space(0, 0, 10, 10, 1000, 1000, "canvas") is True

    def test_forced_frame(self):
        from people_analytics_service.yolo_backend import _row_is_canvas_space

        # Even coordinates that overflow both axes must be treated as
        # already-frame-space once explicitly forced.
        assert _row_is_canvas_space(0, 0, 2000, 2000, 100, 100, "frame") is False

    def test_auto_landscape_vertical_overflow_detected(self):
        """Wide/landscape source frame -> canvas padding is vertical (top/bottom);
        content coordinates can overflow past frame_h. Pre-existing case, must
        keep working after the widened check.
        """
        from people_analytics_service.yolo_backend import _row_is_canvas_space

        assert _row_is_canvas_space(50, 450, 300, 480, frame_w=640, frame_h=400, coord_space="auto") is True

    def test_auto_portrait_horizontal_overflow_detected(self):
        """P1-3 regression: portrait source frame -> canvas padding is
        horizontal (left/right); content coordinates overflow past frame_w
        while staying within frame_h. The original vertical-only heuristic
        (``ry2 > frame_h``) missed this and left the box un-rescaled.
        """
        from people_analytics_service.yolo_backend import _row_is_canvas_space

        assert _row_is_canvas_space(350, 100, 450, 300, frame_w=400, frame_h=640, coord_space="auto") is True

    def test_auto_no_overflow_treated_as_frame_space(self):
        from people_analytics_service.yolo_backend import _row_is_canvas_space

        assert _row_is_canvas_space(10, 10, 50, 50, frame_w=640, frame_h=640, coord_space="auto") is False


# ---------------------------------------------------------------------------
# _parse_end2end (P1-3: full coordinate-recovery pipeline, portrait + landscape)
# ---------------------------------------------------------------------------

class TestParseEnd2End:
    """_parse_end2end doesn't touch `self`, so we can call it unbound without
    constructing a real _LeanPool (which requires an ONNX session)."""

    def _parse(self, out, conf, classes, frame_h, frame_w, r, top, left):
        from people_analytics_service.yolo_backend import _LeanPool

        return _LeanPool._parse_end2end(None, out, conf, classes, frame_h, frame_w, r, top, left)

    def test_landscape_frame_vertical_padding_rescaled(self):
        # 640x400 frame letterboxed onto a 640x640 canvas: r=1.0, top=120, left=0.
        # A canvas-space box at y=[420, 520] must map back to frame y=[300, 400].
        row = [50.0, 420.0, 150.0, 520.0, 0.9, 0.0]
        out = np.array([[row]], dtype=np.float32)
        result = self._parse(out, conf=0.5, classes={0}, frame_h=400, frame_w=640, r=1.0, top=120, left=0)
        assert result.boxes is not None and len(result.boxes) == 1
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
        assert (x1, y1, x2, y2) == pytest.approx((50.0, 300.0, 150.0, 400.0), abs=1.0)

    def test_portrait_frame_horizontal_padding_rescaled(self):
        """P1-3 regression: 400x640 frame letterboxed onto a 640x640 canvas
        (r=1.0, top=0, left=120). A canvas-space box straddling the padding
        (x=[70, 450]) must map back correctly using the left offset -- the
        pre-fix vertical-only heuristic would have skipped this rescale
        entirely (ry2=300 never exceeds frame_h=640), leaving the box shifted
        left by 120px in frame coordinates.
        """
        row = [70.0, 100.0, 450.0, 300.0, 0.9, 0.0]
        out = np.array([[row]], dtype=np.float32)
        result = self._parse(out, conf=0.5, classes={0}, frame_h=640, frame_w=400, r=1.0, top=0, left=120)
        assert result.boxes is not None and len(result.boxes) == 1
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
        # (70-120)=-50 clamped to 0; (450-120)=330.
        assert (x1, y1, x2, y2) == pytest.approx((0.0, 100.0, 330.0, 300.0), abs=1.0)

    def test_already_frame_space_left_untouched(self):
        row = [10.0, 20.0, 100.0, 200.0, 0.9, 0.0]
        out = np.array([[row]], dtype=np.float32)
        result = self._parse(out, conf=0.5, classes={0}, frame_h=640, frame_w=640, r=1.0, top=0, left=0)
        assert result.boxes is not None and len(result.boxes) == 1
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
        assert (x1, y1, x2, y2) == pytest.approx((10.0, 20.0, 100.0, 200.0), abs=1.0)
