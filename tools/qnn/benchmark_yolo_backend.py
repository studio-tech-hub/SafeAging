#!/usr/bin/env python3
"""Benchmark YOLO inference: CPU vs Qualcomm QNN (HTP/GPU).

Usage:
  YOLO_BACKEND=cpu python tools/qnn/benchmark_yolo_backend.py --image tools/_test_frame.jpg
  YOLO_BACKEND=qnn_htp QNN_BACKEND_PATH=/opt/qairt/lib/aarch64-android/libQnnHtp.so \\
      python tools/qnn/benchmark_yolo_backend.py --image tools/_test_frame.jpg --runs 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

os.environ.setdefault("MODEL_PATH", str(REPO_ROOT / "models" / "yolo26n.onnx"))
os.environ.setdefault("DATABASE_URL", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark YOLO backend")
    parser.add_argument("--image", required=True, help="JPEG/PNG test frame")
    parser.add_argument("--runs", type=int, default=10, help="Timed inference runs after warmup")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs (not timed)")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Image not found: {img_path}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Failed to read image: {img_path}", file=sys.stderr)
        return 1

    from people_analytics_service import yolo_backend
    from people_analytics_service.config import MODEL_PATH, YOLO_BACKEND

    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"YOLO_BACKEND={YOLO_BACKEND}")
    print(f"qnn_status={yolo_backend.qnn_runtime_status()}")

    model = yolo_backend.load_yolo_model(MODEL_PATH)
    active = yolo_backend.active_backend()
    print(f"active_backend={active}")

    fh, fw = frame.shape[:2]
    print(f"frame={fw}x{fh} (P1-3: compare against box coord ranges below to sanity-check coord space)")

    for _ in range(args.warmup):
        yolo_backend.predict_yolo(model, frame, conf=0.35, imgsz=640, verbose=False)

    times_ms: list[float] = []
    last_boxes: list[list[float]] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        result = yolo_backend.predict_yolo(model, frame, conf=0.35, imgsz=640, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        boxes = getattr(result, "boxes", None)
        last_boxes = [list(b.xyxy[0].tolist()) for b in boxes] if boxes else []

    arr = np.array(times_ms)
    print(
        f"runs={args.runs} backend={active} "
        f"mean={arr.mean():.1f}ms p50={np.percentile(arr, 50):.1f}ms "
        f"p95={np.percentile(arr, 95):.1f}ms min={arr.min():.1f}ms max={arr.max():.1f}ms"
    )

    # P1-3 box-accuracy sanity check: boxes should span roughly the same pixel
    # range as the input frame. If every box coordinate stays well inside
    # [0, 640) while the frame is much larger than 640x640, that's a red flag
    # that end2end output is still in letterbox-canvas space and was not
    # rescaled -- see END2END_COORD_SPACE in config.py / .env.example.
    print(f"last_run_detections={len(last_boxes)}")
    if last_boxes:
        xs = [c for box in last_boxes for c in (box[0], box[2])]
        ys = [c for box in last_boxes for c in (box[1], box[3])]
        print(
            f"box_coord_range x=[{min(xs):.1f}, {max(xs):.1f}] "
            f"y=[{min(ys):.1f}, {max(ys):.1f}] (frame is {fw}x{fh})"
        )
        if max(xs) < 640 and max(ys) < 640 and (fw > 700 or fh > 700):
            print(
                "WARNING: all box coordinates stay under 640 despite a much larger frame -- "
                "possible unrescaled canvas-space output. Verify against a known detection's "
                "true pixel position and consider END2END_COORD_SPACE=canvas."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
