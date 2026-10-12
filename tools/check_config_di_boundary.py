#!/usr/bin/env python3
"""Config dependency-injection boundary lint (P2-3).

P2-3's target architecture: config.py's AppConfig/get_config() stay the
single source of truth, but NEW/refactored code (the P2-1 pipeline package
and the P2-2 repository module) must receive config as an explicit,
injected parameter rather than importing config.py's module-level global
mirrors (CONFIDENCE_THRESHOLD, ENABLE_ROI, TRACK_MIN_HITS, ...) directly.
Everything else in the codebase (admin_router.py, outbox_worker.py,
db/dal.py, etc.) is explicitly NOT in scope — it keeps reading globals
indefinitely, per P2-3's "no disruptive global rewrite" design.

This script enforces that boundary for exactly two locations:
  - python/people_analytics_service/pipeline/**/*.py
  - python/people_analytics_service/db/repository.py

A violation is any `from <...>config import X` (relative, any dot-level, or
absolute `people_analytics_service.config`) where `X` is one of config.py's
auto-generated global mirrors — computed dynamically by parsing config.py
itself for `NAME = CONFIG.<attr>` assignments, so this check never goes
stale as AppConfig fields are added/renamed. Importing `logger`, `AppConfig`,
`get_config`, or any other non-mirror name (e.g. `normalize_camera_id`,
`log_config_summary`) from config.py is fine and not flagged.

Run with:
    python tools/check_config_di_boundary.py

Exit code 0 = clean, 1 = violations found.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = REPO_ROOT / "python" / "people_analytics_service" / "config.py"

# Relative to python/people_analytics_service/
_GUARDED_PATHS = (
    "pipeline",
    "db/repository.py",
)

# Any `from X import ...` whose module resolves to config.py is in scope,
# whether written as a relative import (`from ...config import Y`, any
# number of dots) or an absolute one (`from people_analytics_service.config
# import Y`).
_CONFIG_MODULE_NAMES = {"config", "people_analytics_service.config"}


def _config_global_mirrors(config_py: Path) -> set[str]:
    """Return every `NAME` in a top-level `NAME = CONFIG.<attr>` assignment
    in config.py — the auto-generated global mirrors this lint forbids
    importing directly into guarded files."""
    tree = ast.parse(config_py.read_text(encoding="utf-8"), filename=str(config_py))
    mirrors: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "CONFIG"
        ):
            mirrors.add(target.id)
    return mirrors


def _guarded_files() -> list[Path]:
    base = REPO_ROOT / "python" / "people_analytics_service"
    files: list[Path] = []
    for rel in _GUARDED_PATHS:
        target = base / rel
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.is_file():
            files.append(target)
    return files


def _check_file(path: Path, mirrors: set[str]) -> list[str]:
    violations: list[str] = []
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        # Not under REPO_ROOT (e.g. a tmp_path fixture in the test suite for
        # this script itself) — fall back to the raw path for display only.
        rel = path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{rel}: could not parse ({exc})"]

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in _CONFIG_MODULE_NAMES:
            continue
        for alias in node.names:
            if alias.name in mirrors:
                violations.append(
                    f"{rel}:{node.lineno}: imports config global '{alias.name}' "
                    f"directly — pass `config: AppConfig` into the function/method "
                    f"instead (see config.get_config() and P2-3)"
                )
    return violations


def main() -> int:
    if not CONFIG_PY.exists():
        print(f"check_config_di_boundary: config.py not found at {CONFIG_PY}")
        return 1

    mirrors = _config_global_mirrors(CONFIG_PY)
    files = _guarded_files()
    if not files:
        print("check_config_di_boundary: no guarded files found — nothing to check")
        return 0

    all_violations: list[str] = []
    for f in files:
        all_violations.extend(_check_file(f, mirrors))

    if all_violations:
        print("check_config_di_boundary: FAILED — direct config-global imports found in guarded code:\n")
        for v in all_violations:
            print(f"  - {v}")
        print(f"\n{len(all_violations)} violation(s) across {len(files)} file(s) checked.")
        return 1

    print(f"check_config_di_boundary: OK — {len(files)} file(s) checked under pipeline/ and db/repository.py, no violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
