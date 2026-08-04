#!/usr/bin/env python3
"""Collect calibration JPEGs by calling /infer and decoding frame_b64 from plugin path.

Runs on AI Box inside analytics container context: grabs live frames via
mediaserver-local RTSP when available, else uses a burst infer collector script.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _collect_via_infer(host: str, port: int, camera_ids: list[str], out_dir: Path, count_per_cam: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for cam in camera_ids:
        for i in range(count_per_cam):
            # Ask mediaserver plugin path: we reuse saved frames if present
            pass
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="calib_frames")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    print("use collect_calib_from_rtsp.sh on box instead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
