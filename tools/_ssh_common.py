#!/usr/bin/env python3
"""Shared SSH/SFTP helper for AI Box ops scripts (tools/_ssh_*.py).

Security model (P0-1):
  * No credentials are hardcoded anywhere in this repo.
  * Connection parameters (host/user/password-or-key) are read from process
    environment variables, optionally pre-loaded from a local, gitignored
    ``tools/ops.env`` file (see ``tools/ops.env.example``).
  * Scripts refuse to run (fail fast, non-zero exit, clear message) if
    required variables are missing or still hold placeholder values.
  * Host key verification is ENABLED by default (``paramiko.RejectPolicy``).
    First-time connections must be trusted once out-of-band (see
    ``bootstrap_known_host`` docstring below) or explicitly opted out of with
    ``--insecure`` / ``AIBOX_SSH_INSECURE=1`` for lab-only use.

Every ``tools/_ssh_*.py`` script should import from this module instead of
hardcoding ``HOST`` / ``USER`` / ``PASSWORD`` / ``paramiko.AutoAddPolicy()``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import paramiko

REPO_ROOT = Path(__file__).resolve().parent.parent
OPS_ENV_PATH = Path(__file__).resolve().parent / "ops.env"

_PLACEHOLDER_VALUES = {"", "changeme", "change-me", "your-password-here", "root_password_here"}


def _load_ops_env() -> None:
    """Populate os.environ from tools/ops.env without overriding real env vars.

    Mirrors the lightweight ``.env`` loader convention already used by
    ``python/people_analytics_service/config.py`` — simple KEY=VALUE lines,
    '#' comments, no external dependency.
    """
    if not OPS_ENV_PATH.exists():
        return
    for raw_line in OPS_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AiBoxCredentials:
    host: str
    user: str
    password: Optional[str]
    key_path: Optional[str]
    remote_dir: str
    sdk_dir: str
    api_key: str


def get_credentials(*, require_sdk_dir: bool = False) -> AiBoxCredentials:
    """Load AI Box connection parameters from the environment, fail fast if missing.

    Required: AIBOX_HOST, AIBOX_USER, and one of AIBOX_PASSWORD / AIBOX_SSH_KEY_PATH.
    Optional: AIBOX_REMOTE_DIR (default /root/SafeAging), AIBOX_SDK_DIR, AIBOX_API_KEY.
    """
    _load_ops_env()

    host = os.getenv("AIBOX_HOST", "").strip()
    user = os.getenv("AIBOX_USER", "").strip()
    password = os.getenv("AIBOX_PASSWORD", "").strip() or None
    key_path = os.getenv("AIBOX_SSH_KEY_PATH", "").strip() or None
    remote_dir = os.getenv("AIBOX_REMOTE_DIR", "/root/SafeAging").strip() or "/root/SafeAging"
    sdk_dir = os.getenv("AIBOX_SDK_DIR", "").strip()
    api_key = os.getenv("AIBOX_API_KEY", "").strip()

    missing = []
    if not host or host.lower() in _PLACEHOLDER_VALUES:
        missing.append("AIBOX_HOST")
    if not user:
        missing.append("AIBOX_USER")
    if (not password or password.lower() in _PLACEHOLDER_VALUES) and not key_path:
        missing.append("AIBOX_PASSWORD (or AIBOX_SSH_KEY_PATH)")
    if require_sdk_dir and not sdk_dir:
        missing.append("AIBOX_SDK_DIR")

    if missing:
        _fail(
            "Missing required AI Box connection settings: "
            + ", ".join(missing)
            + ".\n\nFix: copy tools/ops.env.example to tools/ops.env and fill in real values, "
            "or export the variables in your shell before running this script.\n"
            "tools/ops.env is gitignored and never committed."
        )

    return AiBoxCredentials(
        host=host,
        user=user,
        password=password,
        key_path=key_path,
        remote_dir=remote_dir,
        sdk_dir=sdk_dir,
        api_key=api_key,
    )


def _fail(message: str) -> None:
    print(f"\n[_ssh_common] ERROR: {message}\n", file=sys.stderr)
    sys.exit(2)


def is_insecure_flag() -> bool:
    """True if the caller opted into disabling host-key verification.

    Either pass ``--insecure`` on the command line or set
    ``AIBOX_SSH_INSECURE=1`` in the environment / ops.env. Lab use only —
    this re-enables MITM-vulnerable AutoAddPolicy behavior.
    """
    return "--insecure" in sys.argv or os.getenv("AIBOX_SSH_INSECURE", "").strip() == "1"


def connect(
    creds: Optional[AiBoxCredentials] = None,
    *,
    timeout: int = 20,
    insecure: Optional[bool] = None,
) -> paramiko.SSHClient:
    """Open a paramiko SSH connection using host-key verification by default.

    On first-ever connection to a box, RejectPolicy will raise
    ``paramiko.SSHException: Server ... not found in known_hosts``. Trust the
    host once, out-of-band, with one of:
        ssh root@<host>                 # accept the fingerprint interactively
        ssh-keyscan -H <host> >> ~/.ssh/known_hosts
    or pass --insecure / set AIBOX_SSH_INSECURE=1 for disposable lab boxes.
    """
    creds = creds or get_credentials()
    if insecure is None:
        insecure = is_insecure_flag()

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    user_known_hosts = Path.home() / ".ssh" / "known_hosts"
    if user_known_hosts.exists():
        client.load_host_keys(str(user_known_hosts))

    if insecure:
        print(
            "!!! WARNING: SSH host-key verification is DISABLED (--insecure). "
            "This accepts any host key and is vulnerable to MITM. "
            "Lab/dev use only — never use against a production box. !!!",
            file=sys.stderr,
        )
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

    connect_kwargs = dict(hostname=creds.host, username=creds.user, timeout=timeout)
    if creds.key_path:
        connect_kwargs["key_filename"] = creds.key_path
    else:
        connect_kwargs["password"] = creds.password

    try:
        client.connect(**connect_kwargs)
    except paramiko.SSHException as exc:
        if "not found in known_hosts" in str(exc) or isinstance(exc, paramiko.BadHostKeyException):
            _fail(
                f"Host key for {creds.host} is not trusted ({exc}).\n"
                "Trust it once with: ssh-keyscan -H "
                f"{creds.host} >> ~/.ssh/known_hosts  (verify the fingerprint out-of-band first)\n"
                "or re-run with --insecure for disposable lab boxes only."
            )
        raise
    return client


def upload_files(
    client: paramiko.SSHClient,
    relative_paths: Iterable[str],
    *,
    creds: Optional[AiBoxCredentials] = None,
    repo_root: Path = REPO_ROOT,
    verbose: bool = True,
) -> None:
    """SFTP-upload repo-relative files to <remote_dir>/<relative_path> on the box."""
    creds = creds or get_credentials()
    sftp = client.open_sftp()
    try:
        for rel in relative_paths:
            src = repo_root / rel
            dst = f"{creds.remote_dir}/{rel.replace(chr(92), '/')}"
            sftp.put(str(src), dst)
            if verbose:
                print(f"uploaded {rel}")
    finally:
        sftp.close()


def run(
    client: paramiko.SSHClient,
    command: str,
    *,
    timeout: int = 120,
    stream: bool = True,
) -> tuple[int, bytes, bytes]:
    """Execute a remote command, optionally streaming combined output to stdout."""
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read()
    err = stderr.read()
    code = stdout.channel.recv_exit_status()
    if stream:
        sys.stdout.buffer.write(out)
        sys.stdout.buffer.write(err)
    return code, out, err
