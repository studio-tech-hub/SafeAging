"""Unit tests for tools/generate_secrets.py (P0-3, P0-4).

Run with:
    pytest tools/test_generate_secrets.py -v
"""
from __future__ import annotations

import os
import stat

import pytest

import generate_secrets as gs


class TestGenerateSecrets:
    def test_creates_env_file_with_all_secrets_when_missing(self, tmp_path):
        env_path = tmp_path / ".env"
        assert gs.main(["--env-file", str(env_path)]) == 0

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        for key in ("API_KEY", "POSTGRES_PASSWORD", "MINIO_ROOT_PASSWORD", "GRAFANA_ADMIN_PASSWORD"):
            assert values.get(key), f"{key} was not generated"
            assert values[key].lower() not in gs._PLACEHOLDER_VALUES

    def test_does_not_clobber_existing_real_value_without_force(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("API_KEY=already-a-real-looking-secret-value\n", encoding="utf-8")

        gs.main(["--env-file", str(env_path)])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert values["API_KEY"] == "already-a-real-looking-secret-value"

    def test_replaces_known_placeholder_even_without_force(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("API_KEY=change-me-to-a-strong-secret\n", encoding="utf-8")

        gs.main(["--env-file", str(env_path)])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert values["API_KEY"] != "change-me-to-a-strong-secret"

    def test_force_regenerates_already_real_values(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("API_KEY=already-a-real-looking-secret-value\n", encoding="utf-8")

        gs.main(["--env-file", str(env_path), "--force"])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert values["API_KEY"] != "already-a-real-looking-secret-value"

    def test_syncs_s3_credentials_with_minio_root(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("MINIO_ROOT_USER=safeaging\n", encoding="utf-8")

        gs.main(["--env-file", str(env_path)])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert values["S3_ACCESS_KEY"] == "safeaging"
        assert values["S3_SECRET_KEY"] == values["MINIO_ROOT_PASSWORD"]

    def test_syncs_database_url_password_segment(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text(
            "DATABASE_URL=postgresql://safeaging:old_weak_password@localhost:15432/safeaging\n",
            encoding="utf-8",
        )

        gs.main(["--env-file", str(env_path)])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert "old_weak_password" not in values["DATABASE_URL"]
        assert values["POSTGRES_PASSWORD"] in values["DATABASE_URL"]
        assert values["DATABASE_URL"].startswith("postgresql://safeaging:")
        assert "@localhost:15432/safeaging" in values["DATABASE_URL"]

    def test_sets_api_key_required_true_when_absent(self, tmp_path):
        env_path = tmp_path / ".env"

        gs.main(["--env-file", str(env_path)])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert values["API_KEY_REQUIRED"] == "true"

    def test_does_not_override_explicit_api_key_required_choice(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("API_KEY_REQUIRED=false\n", encoding="utf-8")

        gs.main(["--env-file", str(env_path)])

        values = gs._parse_env(env_path.read_text(encoding="utf-8"))
        assert values["API_KEY_REQUIRED"] == "false"

    @pytest.mark.skipif(os.name != "posix", reason="chmod semantics are POSIX-only (P0-4)")
    def test_hardens_env_file_permissions_to_600(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("API_KEY=already-a-real-looking-secret-value\n", encoding="utf-8")
        env_path.chmod(0o644)  # start from a permissive mode to prove it's tightened

        gs.main(["--env-file", str(env_path)])

        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600

    def test_harden_permissions_is_a_no_op_on_non_posix(self, tmp_path, monkeypatch):
        """On Windows, chmod bits don't map onto NTFS ACLs — must not raise."""
        monkeypatch.setattr(gs.os, "name", "nt")
        env_path = tmp_path / ".env"
        env_path.write_text("x", encoding="utf-8")
        gs._harden_permissions(env_path)  # must not raise
