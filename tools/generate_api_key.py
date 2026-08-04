#!/usr/bin/env python3
"""Generate a cryptographically strong API_KEY for the analytics service.

Usage:
    python tools/generate_api_key.py
    python tools/generate_api_key.py --length 48
    python tools/generate_api_key.py --set-in .env      # writes/updates API_KEY= line in place

The generated value is safe to paste directly into .env as:
    API_KEY=<value>
    API_KEY_REQUIRED=true

For an AI Box deployment, the SAME value must also be set as the
"Service API Key" plugin setting on every camera in Nx Client (see
config/manifest.json's service_api_key setting) — otherwise the plugin will
get HTTP 401 and cameras will stop showing detections.
"""
from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path


def generate_key(length: int = 32) -> str:
    """Return a URL-safe random key with >= length bytes of entropy."""
    return secrets.token_urlsafe(length)


def _set_in_env_file(env_path: Path, key: str) -> None:
    line_re = re.compile(r"^API_KEY=.*$", re.MULTILINE)
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        if line_re.search(text):
            text = line_re.sub(f"API_KEY={key}", text, count=1)
        else:
            sep = "" if text.endswith("\n") or not text else "\n"
            text = text + sep + f"API_KEY={key}\n"
    else:
        text = f"API_KEY={key}\n"
    env_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--length", type=int, default=32, help="random bytes of entropy (default 32)")
    parser.add_argument(
        "--set-in",
        metavar="ENV_FILE",
        help="write/update API_KEY=<value> directly in the given .env file instead of printing to stdout",
    )
    args = parser.parse_args()

    key = generate_key(args.length)

    if args.set_in:
        env_path = Path(args.set_in)
        _set_in_env_file(env_path, key)
        print(f"Wrote API_KEY to {env_path}")
        print("Remember to also set API_KEY_REQUIRED=true in the same file.")
    else:
        print(key)
        print("\nAdd to your .env:", file=sys.stderr)
        print(f"  API_KEY={key}", file=sys.stderr)
        print("  API_KEY_REQUIRED=true", file=sys.stderr)
        print(
            "\nFor an AI Box / Nx plugin deployment, set the SAME value as the "
            "'Service API Key' setting on every camera in Nx Client before "
            "enabling API_KEY_REQUIRED=true, or detections will stop (HTTP 401).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
