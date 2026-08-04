#!/usr/bin/env python3
"""Validate QDQ ONNX vs float: confidences must not be all-zero on calibration set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


def _score_stats(model_path: Path, calib_dir: Path, imgsz: int, max_images: int) -> dict:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    images = sorted(calib_dir.glob("*.jpg"))[:max_images]
    max_conf = 0.0
    nonzero = 0
    tops: list[float] = []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        raw = sess.run(None, {inp: _letterbox(img, imgsz)})[0]
        if raw.ndim == 3 and raw.shape[-1] >= 5:
            scores = raw[0][:, 4]
        else:
            scores = raw.reshape(-1)
        mx = float(np.max(scores))
        max_conf = max(max_conf, mx)
        nonzero += int(np.count_nonzero(scores))
        tops.append(mx)
    return {
        "images": len(images),
        "max_conf": max_conf,
        "nonzero": nonzero,
        "mean_top": float(np.mean(tops)) if tops else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--float-model", required=True)
    parser.add_argument("--qdq-model", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--min-max-conf", type=float, default=0.01)
    args = parser.parse_args()

    calib = Path(args.calib_dir)
    if not calib.is_dir():
        print(f"calib dir missing: {calib}", file=sys.stderr)
        return 1

    fl = _score_stats(Path(args.float_model), calib, args.imgsz, args.max_images)
    qd = _score_stats(Path(args.qdq_model), calib, args.imgsz, args.max_images)
    print("float", fl)
    print("qdq  ", qd)

    if qd["max_conf"] < args.min_max_conf:
        print(
            f"FAIL: QDQ max_conf {qd['max_conf']:.6f} < {args.min_max_conf} (broken quant)",
            file=sys.stderr,
        )
        return 1
    if fl["max_conf"] > 0.05 and qd["max_conf"] < fl["max_conf"] * 0.05:
        print(
            f"WARN: QDQ max_conf much lower than float ({qd['max_conf']:.4f} vs {fl['max_conf']:.4f})",
            file=sys.stderr,
        )
    print("OK: QDQ model produces non-trivial confidences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
