#!/usr/bin/env python3
"""Benchmark P1-6's adaptive face-detector sizing.

Measures latency + detection reliability across a range of real person-crop
sizes, comparing the adaptive bucket (`face_engine.select_det_size()`)
against always using the ceiling (`FACE_DET_SIZE`, the "before P1-6"
behavior) on the exact same crops -- an apples-to-apples before/after
comparison.

This is the deployment-checklist item from CPU_PRODUCTION_PROFILE.md's P1-6
section ("Validation still to do on real hardware"): measure average
face-recognition CPU time/latency per call across a range of real crop
sizes, before vs. after the adaptive-sizing change, and confirm no
regression in match accuracy (here: detection success as a proxy) for
small-but-still-identifiable faces.

Usage (inside the analytics container / on the AI Box, where insightface +
the buffalo_s model pack are already installed):
    python tools/benchmark_face_det_size.py
    python tools/benchmark_face_det_size.py --image tools/_test_frame.jpg --runs 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

os.environ.setdefault("DATABASE_URL", "")

# Simulated person-crop long-side sizes (px), spanning "distant person on a
# wide-angle camera" up to "close-up" -- the same range the adaptive buckets
# (320, 480) vs. the ceiling (640 by default) are meant to cover.
CROP_LONG_SIDES = [160, 240, 320, 400, 480, 560, 640, 800, 960]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark P1-6 adaptive face det_size")
    parser.add_argument(
        "--image", default=str(REPO_ROOT / "tools" / "_test_frame.jpg"),
        help="Real photo containing at least one detectable face",
    )
    parser.add_argument("--runs", type=int, default=10, help="Timed runs per crop size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs (not timed)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to read image: {args.image}", file=sys.stderr)
        return 1
    h0, w0 = img.shape[:2]
    aspect = w0 / h0

    from people_analytics_service import face_engine
    from people_analytics_service.config import FACE_DET_SIZE

    app = face_engine._get_app()
    print(f"FACE_DET_SIZE (ceiling)={FACE_DET_SIZE}  buckets={face_engine._DET_SIZE_BUCKETS}")
    print(f"source image: {args.image} ({w0}x{h0})")
    print()
    header = (
        f"{'long_side':>9} {'crop_wxh':>11} {'adaptive':>8} "
        f"{'adapt_ms':>9} {'ceil_ms':>8} {'speedup':>8} "
        f"{'adapt_hit':>9} {'ceil_hit':>8}"
    )
    print(header)
    print("-" * len(header))

    rows: list[tuple] = []
    for long_side in CROP_LONG_SIDES:
        if aspect >= 1.0:
            cw, ch = long_side, max(1, int(round(long_side / aspect)))
        else:
            ch, cw = long_side, max(1, int(round(long_side * aspect)))
        crop = cv2.resize(img, (cw, ch))

        adaptive_size = face_engine.select_det_size(cw, ch)

        # Warm up both paths (SCRFD caches an ORT session per input size, so
        # the first call at a new size pays a one-time setup cost we don't
        # want polluting the timed average).
        for _ in range(args.warmup):
            face_engine._detect_faces_with_size(app, crop, adaptive_size)
            face_engine._detect_faces_with_size(app, crop, FACE_DET_SIZE)

        adapt_times: list[float] = []
        ceil_times: list[float] = []
        adapt_found = 0
        ceil_found = 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            faces_a = face_engine._detect_faces_with_size(app, crop, adaptive_size)
            adapt_times.append((time.perf_counter() - t0) * 1000.0)
            if faces_a:
                adapt_found += 1

            t0 = time.perf_counter()
            faces_c = face_engine._detect_faces_with_size(app, crop, FACE_DET_SIZE)
            ceil_times.append((time.perf_counter() - t0) * 1000.0)
            if faces_c:
                ceil_found += 1

        a_ms = float(np.mean(adapt_times))
        c_ms = float(np.mean(ceil_times))
        speedup = c_ms / a_ms if a_ms > 0 else float("nan")
        rows.append((long_side, cw, ch, adaptive_size, a_ms, c_ms, speedup, adapt_found, ceil_found))
        print(
            f"{long_side:>9} {f'{cw}x{ch}':>11} {adaptive_size:>8} "
            f"{a_ms:>7.1f}ms {c_ms:>6.1f}ms {speedup:>7.2f}x "
            f"{adapt_found:>6}/{args.runs} {ceil_found:>5}/{args.runs}"
        )

    print()
    below_ceiling = [r for r in rows if r[3] < FACE_DET_SIZE]
    if below_ceiling:
        mean_speedup = float(np.mean([r[6] for r in below_ceiling]))
        regressions = [r for r in below_ceiling if r[7] < r[8]]
        print(f"Mean speedup on crops below the ceiling: {mean_speedup:.2f}x")
        if regressions:
            print(
                f"WARNING: {len(regressions)} crop size(s) had a lower detection-hit "
                f"rate at the adaptive size than at the ceiling -- inspect "
                f"_DET_SIZE_BUCKETS for those long_side values: "
                f"{[r[0] for r in regressions]}"
            )
        else:
            print("No accuracy regression at the adaptive size vs. the ceiling.")
    else:
        print("No crop size fell below the ceiling for this image's aspect ratio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
