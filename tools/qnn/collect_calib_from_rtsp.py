#!/usr/bin/env python3
"""Capture calibration JPEGs from deployed camera RTSP streams (runs on AI Box)."""

from __future__ import annotations

import argparse
import sqlite3
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _streams_from_nx_db(db_path: Path) -> list[tuple[str, str]]:
    """Return (label, rtsp_url) pairs preferring secondary/sub streams."""
    if not db_path.is_file():
        return []
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    rows = cur.execute(
        "SELECT r.name, k.value FROM vms_kvpair k "
        "JOIN vms_resource r ON r.guid = k.resource_guid "
        "WHERE k.key = 'streamUrls'"
    ).fetchall()
    con.close()
    out: list[tuple[str, str]] = []
    for name, raw in rows:
        try:
            urls = json.loads(raw)
        except Exception:
            continue
        if not isinstance(urls, dict):
            continue
        # Prefer encoder 2 / secondary (smaller, matches plugin substream).
        url = urls.get("2") or urls.get("1")
        if url and str(url).startswith("rtsp"):
            out.append((str(name), str(url)))
    return out


def _letterbox_save(img, path: Path, size: int) -> None:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    cv2.imwrite(str(path), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect QNN calibration frames from RTSP")
    parser.add_argument("--out", default="calib_frames", help="Output directory")
    parser.add_argument("--per-stream", type=int, default=16, help="Frames per camera")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument(
        "--nx-db",
        default="/opt/networkoptix/mediaserver/var/ecs.sqlite",
        help="Nx Media Server sqlite DB",
    )
    parser.add_argument("--rtsp", action="append", default=[], help="Extra rtsp:// URLs (label=url)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    streams: list[tuple[str, str]] = _streams_from_nx_db(Path(args.nx_db))
    for item in args.rtsp:
        if "=" in item:
            label, url = item.split("=", 1)
            streams.append((label.strip(), url.strip()))
        else:
            streams.append((f"extra_{len(streams)}", item.strip()))

    if len(streams) < 1:
        print("No RTSP streams found", file=sys.stderr)
        return 1

    saved = 0
    for label, url in streams:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        print(f"[capture] {label} -> {url}")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[warn] cannot open {label}", file=sys.stderr)
            continue
        # Flush a few buffered frames.
        for _ in range(5):
            cap.read()
        got = 0
        while got < args.per_stream:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[warn] read fail {label} at {got}", file=sys.stderr)
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(url)
                continue
            path = out_dir / f"{safe}_{got:03d}.jpg"
            cv2.imwrite(str(path), frame)
            got += 1
            saved += 1
            time.sleep(max(0.05, args.interval))
        cap.release()
        print(f"[ok] {label}: {got} frames")

    print(f"[done] saved {saved} frames -> {out_dir}")
    return 0 if saved >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
