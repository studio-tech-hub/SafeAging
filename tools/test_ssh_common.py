"""Unit tests for tools/_ssh_common.py (P0-1: no hardcoded ops credentials).

Run with:
    pytest tools/test_ssh_common.py -v

These tests only exercise credential loading/validation — they never open a
real network connection.
"""
from __future__ import annotations

import importlib

import pytest

import _ssh_common as ssh_common

AIBOX_ENV_VARS = [
    "AIBOX_HOST",
    "AIBOX_USER",
    "AIBOX_PASSWORD",
    "AIBOX_SSH_KEY_PATH",
    "AIBOX_REMOTE_DIR",
    "AIBOX_SDK_DIR",
    "AIBOX_API_KEY",
    "AIBOX_SSH_INSECURE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Ensure no real ops credentials leak into the test environment."""
    for var in AIBOX_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Point the module at an ops.env path that does not exist, so tests are
    # hermetic regardless of whether a real tools/ops.env is present on disk.
    monkeypatch.setattr(ssh_common, "OPS_ENV_PATH", tmp_path / "ops.env")
    yield


def test_get_credentials_fails_fast_when_all_missing(capsys):
    with pytest.raises(SystemExit) as exc_info:
        ssh_common.get_credentials()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "AIBOX_HOST" in err
    assert "AIBOX_USER" in err
    assert "AIBOX_PASSWORD" in err
    assert "ops.env" in err


def test_get_credentials_fails_fast_on_placeholder_values(monkeypatch, capsys):
    monkeypatch.setenv("AIBOX_HOST", "192.168.1.210")
    monkeypatch.setenv("AIBOX_USER", "root")
    monkeypatch.setenv("AIBOX_PASSWORD", "changeme")
    with pytest.raises(SystemExit) as exc_info:
        ssh_common.get_credentials()
    assert exc_info.value.code == 2
    assert "AIBOX_PASSWORD" in capsys.readouterr().err


def test_get_credentials_succeeds_with_password(monkeypatch):
    monkeypatch.setenv("AIBOX_HOST", "10.0.0.5")
    monkeypatch.setenv("AIBOX_USER", "root")
    monkeypatch.setenv("AIBOX_PASSWORD", "a-real-secret")
    creds = ssh_common.get_credentials()
    assert creds.host == "10.0.0.5"
    assert creds.user == "root"
    assert creds.password == "a-real-secret"
    assert creds.key_path is None
    assert creds.remote_dir == "/root/SafeAging"  # documented default


def test_get_credentials_succeeds_with_ssh_key(monkeypatch):
    monkeypatch.setenv("AIBOX_HOST", "10.0.0.5")
    monkeypatch.setenv("AIBOX_USER", "root")
    monkeypatch.setenv("AIBOX_SSH_KEY_PATH", "/home/dev/.ssh/id_ed25519")
    creds = ssh_common.get_credentials()
    assert creds.password is None
    assert creds.key_path == "/home/dev/.ssh/id_ed25519"


def test_get_credentials_requires_sdk_dir_when_requested(monkeypatch, capsys):
    monkeypatch.setenv("AIBOX_HOST", "10.0.0.5")
    monkeypatch.setenv("AIBOX_USER", "root")
    monkeypatch.setenv("AIBOX_PASSWORD", "a-real-secret")
    with pytest.raises(SystemExit):
        ssh_common.get_credentials(require_sdk_dir=True)
    assert "AIBOX_SDK_DIR" in capsys.readouterr().err

    monkeypatch.setenv("AIBOX_SDK_DIR", "/root/metadata_sdk")
    creds = ssh_common.get_credentials(require_sdk_dir=True)
    assert creds.sdk_dir == "/root/metadata_sdk"


def test_load_ops_env_does_not_override_real_env_vars(monkeypatch, tmp_path):
    ops_env = tmp_path / "ops.env"
    ops_env.write_text("AIBOX_HOST=1.2.3.4\nAIBOX_USER=fromfile\n", encoding="utf-8")
    monkeypatch.setattr(ssh_common, "OPS_ENV_PATH", ops_env)
    monkeypatch.setenv("AIBOX_HOST", "9.9.9.9")  # real env wins
    monkeypatch.setenv("AIBOX_PASSWORD", "a-real-secret")

    creds = ssh_common.get_credentials()
    assert creds.host == "9.9.9.9"
    assert creds.user == "fromfile"


def test_no_hardcoded_credentials_in_module_source():
    """Regression guard: the shared helper itself must never embed real secrets."""
    source = importlib.import_module("_ssh_common").__file__
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()
    assert "oelinux123" not in text
    assert "192.168.1.210" not in text
