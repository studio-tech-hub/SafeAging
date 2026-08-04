#!/usr/bin/env python3
"""Quantize YOLO26 ONNX to a QNN-friendly QDQ model.

This follows ONNX Runtime's QNN quantization flow:
  1. preprocess/simplify the ONNX graph for QNN,
  2. calibrate with real camera frames,
  3. emit a QDQ ONNX model suitable for QNN HTP.

Example:
  python tools/qnn/export_yolo26_qdq.py \
    --input models/yolo26n.onnx \
    --calib-dir calib_frames \
    --output models/yolo26n_qdq.onnx
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(Path(p) for p in glob.glob(str(root / "**" / ext), recursive=True))
    return sorted({p.resolve() for p in paths})


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def _preprocess(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    lb = _letterbox(img, size)
    x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: Iterable[Path], input_name: str, imgsz: int) -> None:
        self._paths = list(image_paths)
        self._input_name = input_name
        self._imgsz = imgsz
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._idx < len(self._paths):
            path = self._paths[self._idx]
            self._idx += 1
            try:
                return {self._input_name: _preprocess(path, self._imgsz)}
            except Exception as exc:
                print(f"[warn] skip {path}: {exc}", file=sys.stderr)
        return None


class SyntheticCalibrationReader(CalibrationDataReader):
    def __init__(self, count: int, input_name: str, imgsz: int) -> None:
        self._count = count
        self._input_name = input_name
        self._imgsz = imgsz
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= self._count:
            return None
        self._idx += 1
        # Smoke-test only: use smooth random camera-like input to validate QDQ/HTP compatibility.
        img = np.random.default_rng(self._idx).integers(
            16, 224, size=(self._imgsz, self._imgsz, 3), dtype=np.uint8
        )
        img = cv2.GaussianBlur(img, (5, 5), 0)
        x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return {self._input_name: np.ascontiguousarray(x[None])}


def _quant_type(raw: str) -> QuantType:
    value = raw.lower()
    if value in {"uint8", "quint8", "u8"}:
        return QuantType.QUInt8
    if value in {"uint16", "quint16", "u16"}:
        return QuantType.QUInt16
    if value in {"int8", "qint8", "i8"}:
        return QuantType.QInt8
    raise argparse.ArgumentTypeError(f"unsupported quant type: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export QNN QDQ ONNX for YOLO26")
    parser.add_argument("--input", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output QDQ ONNX model")
    parser.add_argument("--calib-dir", help="Directory of real camera images")
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=0,
        help="Use synthetic calibration samples for HTP smoke tests only",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO square input size")
    parser.add_argument("--max-images", type=int, default=128, help="Max calibration images")
    parser.add_argument(
        "--activation-type",
        type=_quant_type,
        default=QuantType.QUInt16,
        help="Activation type: uint16 (default), uint8, or int8",
    )
    parser.add_argument(
        "--weight-type",
        type=_quant_type,
        default=QuantType.QUInt8,
        help="Weight type: uint8 (default) or int8",
    )
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel weights")
    parser.add_argument("--keep-preprocessed", action="store_true", help="Keep intermediate preprocessed ONNX")
    args = parser.parse_args()

    input_model = Path(args.input)
    output_model = Path(args.output)
    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    if not input_model.is_file():
        print(f"Input model not found: {input_model}", file=sys.stderr)
        return 1
    if args.synthetic_count <= 0 and (calib_dir is None or not calib_dir.is_dir()):
        print(f"Calibration directory not found: {calib_dir}", file=sys.stderr)
        return 1

    images: list[Path] = []
    if args.synthetic_count <= 0:
        images = _images(calib_dir)[: max(1, args.max_images)]
        if len(images) < 8:
            print(f"Need at least 8 calibration images, found {len(images)}", file=sys.stderr)
            return 1

    sess = ort.InferenceSession(str(input_model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"[i] input={input_model} output={output_model}")
    sample_count = args.synthetic_count if args.synthetic_count > 0 else len(images)
    print(f"[i] input_name={input_name} imgsz={args.imgsz} calib_samples={sample_count}")
    print(f"[i] activation={args.activation_type.name} weight={args.weight_type.name}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    preproc_model = output_model.with_name(output_model.stem + "_preproc.onnx")
    try:
        changed = qnn_preprocess_model(str(input_model), str(preproc_model))
    except Exception as exc:
        print(f"[warn] qnn_preprocess_model failed; quantizing original model: {exc}", file=sys.stderr)
        changed = False
    model_to_quantize = preproc_model if changed else input_model
    print(f"[i] preprocessed={changed} model_to_quantize={model_to_quantize}")

    if args.synthetic_count > 0:
        print("[warn] synthetic calibration is for QNN smoke tests only; do not ship this model")
        reader: CalibrationDataReader = SyntheticCalibrationReader(
            args.synthetic_count,
            input_name=input_name,
            imgsz=args.imgsz,
        )
    else:
        reader = ImageCalibrationReader(images, input_name=input_name, imgsz=args.imgsz)
    qnn_config = get_qnn_qdq_config(
        str(model_to_quantize),
        reader,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        per_channel=args.per_channel,
        calibration_providers=["CPUExecutionProvider"],
    )
    quantize(str(model_to_quantize), str(output_model), qnn_config)

    if preproc_model.exists() and not args.keep_preprocessed:
        preproc_model.unlink()

    print(f"[ok] wrote {output_model} ({output_model.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
