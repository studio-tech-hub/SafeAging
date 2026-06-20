#!/usr/bin/env python3
"""Export YOLO26 to ONNX for AI Box deployment (imgsz=640 default).

Usage (Windows dev or Linux box):
    pip install "ultralytics>=8.4.0"
    python tools/export_yolo26_onnx.py
    python tools/export_yolo26_onnx.py --model yolo26n.pt --imgsz 640 --out models/yolo26n.onnx

Ultralytics YOLO() can load .onnx at inference time (onnxruntime backend).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO26 weights to ONNX")
    parser.add_argument(
        "--model",
        default="models/yolo26n.pt",
        help="Source weights (.pt); downloads via Ultralytics if missing",
    )
    parser.add_argument("--out", default="models/yolo26n.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", action="store_false", dest="simplify")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install 'ultralytics>=8.4.0'", file=sys.stderr)
        return 1

    print(f"Loading: {model_path}")
    model = YOLO(str(model_path))

    print(f"Exporting ONNX → {out_path} (imgsz={args.imgsz}, opset={args.opset})")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
    )
    export_path = Path(export_path)
    if export_path.resolve() != out_path.resolve():
        out_path.write_bytes(export_path.read_bytes())
        print(f"Copied to {out_path}")

    print(f"OK: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Set on AI Box: MODEL_PATH=/app/models/yolo26n.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
