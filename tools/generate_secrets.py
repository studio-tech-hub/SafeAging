#!/usr/bin/env python3
"""First-boot helper: generate every required secret in one shot (P0-3).

docker-compose.yml / docker-compose.aibox-hostdb.yml refuse to start unless
API_KEY, POSTGRES_PASSWORD, MINIO_ROOT_PASSWORD, and GRAFANA_ADMIN_PASSWORD
are all set to real (non-empty, non-placeholder) values in .env. This script
generates all of them at once instead of running tools/generate_api_key.py
and inventing three more passwords by hand.

Usage:
    python tools/generate_secrets.py                  # writes/updates ./.env
    python tools/generate_secrets.py --env-file .env  # same, explicit path
    python tools/generate_secrets.py --force          # also replace values
                                                        # that already look real

By default this is idempotent and non-destructive: any variable that already
holds a real-looking (non-empty, non-placeholder) value is left untouched, so
re-running this against an existing installation's .env is safe. Use --force
to rotate every secret unconditionally (e.g. after a suspected compromise —
see also P0-4 secrets hygiene).

S3_ACCESS_KEY / S3_SECRET_KEY are kept in sync with MINIO_ROOT_USER /
MINIO_ROOT_PASSWORD, since the analytics service authenticates to MinIO as
the root user (see .env.example).

For an AI Box / Nx plugin deployment, remember to also set the SAME API_KEY
value as the "Service API Key" setting on every camera in Nx Client — this
script only updates .env, it does not touch Nx Witness configuration.

Secrets hygiene (P0-4): after writing, this script hardens .env to mode 0600
(owner read/write only) on POSIX systems, since it now holds real credentials
instead of dev placeholders. This is a best-effort no-op on Windows (NTFS
ACLs don't map onto POSIX chmod bits) — Windows users should rely on normal
filesystem/user-account isolation, or WSL, for a production deployment.
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import stat
import sys
from pathlib import Path

# Same placeholder set as tools/check_compose_security.py and
# people_analytics_service.config._INSECURE_API_KEY_VALUES — kept in sync by
# hand since these are three small, independent, rarely-changed lists.
_PLACEHOLDER_VALUES = {
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
    "replace_with_postgres_password",
}

# name -> (env var, byte length for secrets.token_urlsafe)
_SECRETS = {
    "API_KEY": 32,
    "POSTGRES_PASSWORD": 24,
    "MINIO_ROOT_PASSWORD": 24,
    "GRAFANA_ADMIN_PASSWORD": 24,
}


def _generate(length: int) -> str:
    return secrets.token_urlsafe(length)


def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _set_var(text: str, key: str, value: str) -> str:
    line_re = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if line_re.search(text):
        return line_re.sub(f"{key}={value}", text, count=1)
    sep = "" if text.endswith("\n") or not text else "\n"
    return text + sep + f"{key}={value}\n"


def _harden_permissions(env_path: Path) -> None:
    """Best-effort chmod 600 on .env (P0-4) — it now holds real credentials.
    Silently skipped on platforms/filesystems where chmod isn't meaningful
    (e.g. Windows, some network filesystems) rather than failing the whole
    secret-generation run over a permissions nicety."""
    if os.name != "posix":
        return
    try:
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError as e:
        print(f"Warning: could not chmod 600 {env_path}: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=".env", help="path to the .env file to write/update (default: .env)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate every secret even if it already holds a real-looking value",
    )
    args = parser.parse_args(argv)

    env_path = Path(args.env_file)
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    current = _parse_env(text)

    generated: dict[str, str] = {}
    skipped: list[str] = []

    for key, length in _SECRETS.items():
        existing = current.get(key, "")
        is_placeholder = existing.lower() in _PLACEHOLDER_VALUES
        if not args.force and existing and not is_placeholder:
            skipped.append(key)
            continue
        value = _generate(length)
        text = _set_var(text, key, value)
        generated[key] = value

    if "MINIO_ROOT_PASSWORD" in generated:
        # The analytics service authenticates to MinIO as the root user —
        # keep S3_ACCESS_KEY/S3_SECRET_KEY in lockstep so it doesn't get
        # locked out after rotation.
        minio_user = current.get("MINIO_ROOT_USER", "safeaging") or "safeaging"
        text = _set_var(text, "S3_ACCESS_KEY", minio_user)
        text = _set_var(text, "S3_SECRET_KEY", generated["MINIO_ROOT_PASSWORD"])

    if "POSTGRES_PASSWORD" in generated:
        # DATABASE_URL (used only for local/direct, non-Docker runs — Compose
        # builds its own from POSTGRES_* automatically) embeds the password
        # inline and would otherwise silently go stale after rotation.
        db_url_re = re.compile(r"^DATABASE_URL=postgresql://([^:@\s]*):[^@\s]*@(.*)$", re.MULTILINE)
        match = db_url_re.search(text)
        if match:
            user, rest = match.group(1), match.group(2)
            text = db_url_re.sub(f"DATABASE_URL=postgresql://{user}:{generated['POSTGRES_PASSWORD']}@{rest}", text, count=1)

    if "API_KEY_REQUIRED" not in current:
        text = _set_var(text, "API_KEY_REQUIRED", "true")

    env_path.write_text(text, encoding="utf-8")
    _harden_permissions(env_path)

    if generated:
        print(f"Generated {len(generated)} secret(s) in {env_path}:")
        for key in generated:
            print(f"  - {key}")
    if skipped:
        print(f"\nLeft {len(skipped)} already-set secret(s) untouched (use --force to rotate):")
        for key in skipped:
            print(f"  - {key}")
    if not generated and not skipped:
        print(f"Nothing to do — {env_path} already has all required secrets set.")

    if "API_KEY" in generated:
        print(
            "\nIMPORTANT: for an AI Box / Nx plugin deployment, also set the new "
            "API_KEY value as the 'Service API Key' setting on every camera in "
            "Nx Client, or the plugin will get HTTP 401 and cameras will stop "
            "showing detections."
        )
    if "MINIO_ROOT_PASSWORD" in generated or "POSTGRES_PASSWORD" in generated:
        print(
            "\nNote: if postgres/minio containers already have data volumes from a "
            "previous run with the OLD password, they will need to be recreated "
            "(docker compose down -v) or updated in place — restarting with a new "
            "password does not retroactively change already-initialized service data."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
