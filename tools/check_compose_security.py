#!/usr/bin/env python3
"""Pre-flight security lint for docker-compose*.yml files (P0-2, P0-3).

Fails (non-zero exit) if any compose file would ship an insecure default:
  - API_KEY_REQUIRED hardcoded to false (or falling back to false), instead
    of defaulting to true.
  - Any secret in _SECRET_VARS (API_KEY, POSTGRES_PASSWORD,
    MINIO_ROOT_PASSWORD, GRAFANA_ADMIN_PASSWORD, S3_SECRET_KEY):
      * hardcoded as a literal value on its own assignment line, or
      * referenced anywhere on a line (e.g. embedded in a DATABASE_URL, or
        assigned to a differently-named compose var like
        GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}) via
        `${VAR:-weak_default}` or bare `${VAR}`, instead of the required
        (`${VAR:?msg}`) syntax that fails loudly when unset.

This intentionally checks ALL docker-compose*.yml files in the repo (tracked
or not) EXCEPT the ones in _DEV_ONLY_EXEMPT below, so a regression is caught
before it's ever committed, not just in CI. Wire into CI once a pipeline
exists (P1-7):

    python tools/check_compose_security.py

Exit code 0 = clean, 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Known weak/placeholder values that must never be reachable as a *silent*
# fallback default (via `${VAR:-value}`) for a security-sensitive variable.
# Kept in sync by hand with tools/generate_secrets.py's _PLACEHOLDER_VALUES
# and people_analytics_service.config._INSECURE_API_KEY_VALUES.
_WEAK_VALUES = {
    "",
    "changeme",
    "change-me",
    "change_me",
    "change-me-to-a-strong-secret",
    "your-api-key-here",
    "secret",
    "password",
    "admin",
    "test",
    "apikey",
    "safeaging_dev_password",
    "safeaging_minio_password",
}

# Vars that must always be interpolated with required (`${VAR:?msg}`) syntax,
# never a hardcoded literal, a bare `${VAR}`, or a `${VAR:-default}` fallback
# (P0-2: API_KEY; P0-3: the rest).
_SECRET_VARS = {
    "API_KEY",
    "POSTGRES_PASSWORD",
    "MINIO_ROOT_PASSWORD",
    "GRAFANA_ADMIN_PASSWORD",
    "S3_SECRET_KEY",
}
_FALSY = {"false", "0", "no"}

# Files that are intentionally local-development-only convenience stacks
# (no `analytics` service, not referenced by any deployment doc/script, not
# reachable by a customer install) and are therefore allowed to keep soft
# defaults so `docker compose up` works with zero config on a laptop. This is
# the "docker-compose.dev.yml" carve-out from the P0-3 plan — reusing the
# existing infra/dev stack instead of adding a second, harder-to-maintain
# duplicate, since Compose cannot satisfy a `${VAR:?msg}` required in one
# file via an override in another (interpolation happens per-file before merge).
_DEV_ONLY_EXEMPT = {"infra/dev/docker-compose.yml"}


def _compose_files() -> list[Path]:
    all_files = (p for p in REPO_ROOT.rglob("docker-compose*.yml") if ".git" not in p.parts)
    return sorted(p for p in all_files if p.relative_to(REPO_ROOT).as_posix() not in _DEV_ONLY_EXEMPT)


def _strip_quotes(value: str) -> str:
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    return value


def _assignment_value(key: str, stripped_line: str) -> str | None:
    """If `stripped_line` is an assignment TO exactly `key` (list style
    `- KEY=value` or mapping style `KEY: value`), return the value part
    (quotes stripped). Otherwise None."""
    if stripped_line.startswith("-"):
        content = _strip_quotes(stripped_line[1:].strip())
        if content.startswith(f"{key}="):
            return content[len(key) + 1 :]
        return None
    prefix = f"{key}:"
    if stripped_line.startswith(prefix):
        return _strip_quotes(stripped_line[len(prefix) :].strip())
    return None


def _check_api_key_required(line: str, rel: Path, lineno: int, violations: list[str]) -> None:
    stripped = line.strip()
    value = _assignment_value("API_KEY_REQUIRED", stripped)
    if value is None:
        return
    if not value.startswith("${") and value.lower() in _FALSY:
        violations.append(
            f"{rel}:{lineno}: API_KEY_REQUIRED hardcoded to '{value}' — "
            f'must default to true, e.g. "API_KEY_REQUIRED=${{API_KEY_REQUIRED:-true}}"'
        )
    fallback = re.match(r"\$\{API_KEY_REQUIRED:-(?P<default>[^}]*)\}$", value)
    if fallback and fallback.group("default").strip().lower() in _FALSY:
        violations.append(
            f"{rel}:{lineno}: API_KEY_REQUIRED fallback default is "
            f"'{fallback.group('default')}' — must default to true"
        )


def _check_hardcoded_literal(key: str, line: str, rel: Path, lineno: int, violations: list[str]) -> None:
    """Flag `KEY=literal` / `KEY: literal` with no `${...}` interpolation
    at all — a secret baked directly into a tracked file."""
    value = _assignment_value(key, line.strip())
    if value is not None and "${" not in value:
        violations.append(
            f"{rel}:{lineno}: {key} is a hardcoded literal in a tracked "
            f'file — use "{key}=${{{key}:?Set {key} in .env}}" instead'
        )


# Matches ${KEY}, ${KEY:-default}, ${KEY-default}, ${KEY:?msg}, ${KEY?msg}
# anywhere in a line, regardless of what compose var it's being assigned to
# (catches e.g. GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}
# and DATABASE_URL=...${POSTGRES_PASSWORD:-weak}@postgres...).
def _interpolation_pattern(key: str) -> re.Pattern[str]:
    # (?![A-Za-z0-9_]) prevents "API_KEY" from matching inside the longer
    # "API_KEY_REQUIRED" — without it, ${API_KEY_REQUIRED:-true} would be
    # misparsed as "${API_KEY}" with modifier=None (a false positive).
    return re.compile(rf"\$\{{{re.escape(key)}(?![A-Za-z0-9_])(:-|-|:\?|\?)?([^}}]*)\}}")


def _check_interpolations(key: str, line: str, rel: Path, lineno: int, violations: list[str]) -> None:
    for match in _interpolation_pattern(key).finditer(line):
        modifier, rest = match.group(1), match.group(2)
        if modifier in (":?", "?"):
            continue  # required syntax — fails loudly if unset, as intended
        if modifier in (":-", "-"):
            if rest.strip().lower() in _WEAK_VALUES:
                violations.append(
                    f"{rel}:{lineno}: {key} silently falls back to placeholder "
                    f"'{rest.strip()}' via '${{{key}{modifier}{rest}}}' — use required "
                    f"('${{{key}:?msg}}') syntax instead so misconfiguration fails loudly"
                )
            continue
        # Bare ${KEY} with no modifier at all — silently becomes "" if unset.
        violations.append(
            f"{rel}:{lineno}: {key} is referenced as '${{{key}}}' with no "
            f"required-value guard — use '${{{key}:?msg}}' instead so a "
            f"missing value fails loudly instead of silently becoming empty"
        )


def _check_file(path: Path) -> list[str]:
    violations: list[str] = []
    rel = path.relative_to(REPO_ROOT)

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        _check_api_key_required(raw_line, rel, lineno, violations)
        for key in _SECRET_VARS:
            _check_hardcoded_literal(key, raw_line, rel, lineno, violations)
            _check_interpolations(key, raw_line, rel, lineno, violations)

    return violations


def main() -> int:
    files = _compose_files()
    if not files:
        print("check_compose_security: no docker-compose*.yml files found — nothing to check")
        return 0

    all_violations: list[str] = []
    for f in files:
        all_violations.extend(_check_file(f))

    if all_violations:
        print("check_compose_security: FAILED — insecure compose defaults found:\n")
        for v in all_violations:
            print(f"  - {v}")
        print(f"\n{len(all_violations)} violation(s) across {len(files)} file(s) checked.")
        return 1

    print(f"check_compose_security: OK — {len(files)} compose file(s) checked, no violations.")
    for f in files:
        print(f"  - {f.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
