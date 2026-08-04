#!/usr/bin/env python3
"""Dynamic pre-flight check for docker-compose*.yml secret requirements (P0-3).

Complements tools/check_compose_security.py (a static text lint) by actually
invoking `docker compose config` twice:

  1. With no .env file at all (a fresh checkout)   -> must FAIL (exit != 0).
  2. With a freshly generated, fully-populated .env -> must SUCCEED, for
     every meaningful compose file combination (base alone, base + AI Box
     overlay, base + AI Box + QNN overlay).

Why this script writes directly to ./.env (temporarily) instead of using
`docker compose --env-file <fixture>`: docker-compose.yml's analytics
service has `env_file: - .env`, which requires a literal ./.env file to
exist relative to the compose project directory *regardless* of
`--env-file` (that flag only affects `${VAR}` template interpolation, a
separate mechanism) — so a real ./.env must exist for either check to
reflect what actually happens on a fresh checkout / CI runner. This script
backs up any existing ./.env before touching it and ALWAYS restores it
(even on error/Ctrl-C), so it is safe to run against a real local .env with
real secrets already in it.

Wire into CI once a pipeline exists (P1-7):

    python tools/verify_compose_config.py

Requires the `docker` CLI (with Compose v2) to be available; does NOT
require the Docker daemon to be running (`config` only parses/interpolates
YAML, it doesn't talk to the daemon).

Exit code 0 = all checks passed, 1 = a check failed unexpectedly.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
BACKUP_PATH = REPO_ROOT / ".env.verify_compose_config.bak"

_COMBOS: list[list[str]] = [
    ["docker-compose.yml"],
    ["docker-compose.yml", "docker-compose.aibox-hostdb.yml"],
    ["docker-compose.yml", "docker-compose.aibox-hostdb.yml", "docker-compose.aibox-qnn.yml"],
]


def _run_config(compose_files: list[str]) -> subprocess.CompletedProcess[str]:
    args = ["docker", "compose"]
    for f in compose_files:
        args += ["-f", f]
    args += ["config", "--quiet"]
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)


def _backup_real_env() -> bool:
    if ENV_PATH.exists():
        shutil.move(str(ENV_PATH), str(BACKUP_PATH))
        return True
    return False


def _restore_real_env(had_backup: bool) -> None:
    if ENV_PATH.exists():
        ENV_PATH.unlink()
    if had_backup:
        shutil.move(str(BACKUP_PATH), str(ENV_PATH))


def main() -> int:
    failures: list[str] = []
    had_backup = _backup_real_env()

    try:
        print("=== Check 1: no .env file at all (fresh checkout) -> docker compose config must FAIL ===")
        result = _run_config(["docker-compose.yml"])
        if result.returncode == 0:
            failures.append("docker compose config SUCCEEDED with no .env at all — required-var enforcement is broken")
            print("  FAIL (expected non-zero exit, got 0)")
        else:
            last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr"
            print(f"  OK (failed as expected: {last_line})")

        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "generate_secrets.py"), "--force"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        for combo in _COMBOS:
            label = " + ".join(combo)
            print(f"\n=== Check: {label} -> docker compose config must SUCCEED with a fully populated .env ===")
            result = _run_config(combo)
            if result.returncode != 0:
                failures.append(f"{label}: docker compose config FAILED with a fully populated .env:\n{result.stderr}")
                print(f"  FAIL:\n{result.stderr}")
            else:
                print("  OK")
    finally:
        _restore_real_env(had_backup)

    if failures:
        print(f"\nverify_compose_config: FAILED — {len(failures)} check(s) did not behave as expected:\n")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nverify_compose_config: OK — all compose secret-requirement checks passed. Original .env (if any) restored untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
