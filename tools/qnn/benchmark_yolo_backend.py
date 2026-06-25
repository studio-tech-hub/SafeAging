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

    for _ in range(args.warmup):
        yolo_backend.predict_yolo(model, frame, conf=0.35, imgsz=640, verbose=False)

    times_ms: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        yolo_backend.predict_yolo(model, frame, conf=0.35, imgsz=640, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    print(
        f"runs={args.runs} backend={active} "
        f"mean={arr.mean():.1f}ms p50={np.percentile(arr, 50):.1f}ms "
        f"p95={np.percentile(arr, 95):.1f}ms min={arr.min():.1f}ms max={arr.max():.1f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
