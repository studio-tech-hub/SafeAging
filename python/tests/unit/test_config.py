"""Unit tests for fail-closed authentication config (P0-2).

AppConfig reads process env vars at construction time, so we can exercise
the auth validation branches by instantiating it directly with monkeypatched
env vars — no need to reload the whole config module or run a live service.
"""
from __future__ import annotations

import pytest

from people_analytics_service.config import AppConfig, _env_or_file


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Ensure each test starts from a known auth-env state, regardless of
    whatever a real local .env file may have already loaded into os.environ."""
    for var in ("API_KEY_REQUIRED", "API_KEY", "ALLOW_INSECURE_NO_AUTH"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestFailClosedAuth:
    def test_default_requires_api_key(self):
        """API_KEY_REQUIRED defaults to true; a missing API_KEY must now
        refuse to start instead of silently disabling auth (the P0-2 fix)."""
        with pytest.raises(RuntimeError, match="API_KEY_REQUIRED=true"):
            AppConfig()

    def test_placeholder_api_key_is_rejected(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "change-me-to-a-strong-secret")
        with pytest.raises(RuntimeError):
            AppConfig()

    def test_other_known_placeholder_values_are_rejected(self, monkeypatch):
        for placeholder in ("changeme", "admin", "SECRET", "  "):
            monkeypatch.setenv("API_KEY", placeholder)
            with pytest.raises(RuntimeError):
                AppConfig()

    def test_real_api_key_succeeds(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "a-real-random-secret-value")
        cfg = AppConfig()
        assert cfg.api_key_required is True
        assert cfg.api_key == "a-real-random-secret-value"
        assert cfg.auth_effectively_disabled is False

    def test_explicit_api_key_required_false_does_not_raise(self, monkeypatch):
        """Operators may still intentionally disable auth — this must not
        raise, only warn, since it's an explicit (not silent) choice."""
        monkeypatch.setenv("API_KEY_REQUIRED", "false")
        cfg = AppConfig()
        assert cfg.api_key_required is False
        assert cfg.auth_effectively_disabled is True

    def test_escape_hatch_preserves_old_behavior(self, monkeypatch):
        """ALLOW_INSECURE_NO_AUTH=true reproduces the pre-P0-2 auto-disable
        behavior, but only when explicitly opted into."""
        monkeypatch.setenv("API_KEY_REQUIRED", "true")
        monkeypatch.setenv("ALLOW_INSECURE_NO_AUTH", "true")
        cfg = AppConfig()  # must NOT raise
        assert cfg.api_key_required is False
        assert cfg.auth_effectively_disabled is True
        assert cfg.allow_insecure_no_auth is True

    def test_escape_hatch_off_by_default(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "a-real-random-secret-value")
        cfg = AppConfig()
        assert cfg.allow_insecure_no_auth is False

    def test_escape_hatch_alone_without_required_true_is_a_no_op(self, monkeypatch):
        """ALLOW_INSECURE_NO_AUTH=true with API_KEY_REQUIRED=false (already
        insecure) should not error and should not change behavior."""
        monkeypatch.setenv("API_KEY_REQUIRED", "false")
        monkeypatch.setenv("ALLOW_INSECURE_NO_AUTH", "true")
        cfg = AppConfig()
        assert cfg.api_key_required is False
        assert cfg.auth_effectively_disabled is True

    def test_api_key_file_populated_takes_effect(self, monkeypatch, tmp_path):
        """P0-4: API_KEY_FILE (Docker secrets convention) is read into
        CONFIG.api_key, exactly like the official postgres/mysql images'
        `*_PASSWORD_FILE` convention."""
        secret_file = tmp_path / "api_key"
        secret_file.write_text("secret-from-mounted-file\n", encoding="utf-8")
        monkeypatch.setenv("API_KEY_FILE", str(secret_file))
        cfg = AppConfig()
        assert cfg.api_key == "secret-from-mounted-file"  # trailing newline stripped
        assert cfg.auth_effectively_disabled is False

    def test_api_key_file_takes_precedence_over_inline_value(self, monkeypatch, tmp_path):
        secret_file = tmp_path / "api_key"
        secret_file.write_text("from-file-wins", encoding="utf-8")
        monkeypatch.setenv("API_KEY_FILE", str(secret_file))
        monkeypatch.setenv("API_KEY", "from-inline-env-var")
        cfg = AppConfig()
        assert cfg.api_key == "from-file-wins"

    def test_api_key_file_missing_falls_back_to_inline_value_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("API_KEY_FILE", "/nonexistent/path/to/secret")
        monkeypatch.setenv("API_KEY", "a-real-random-secret-value")
        cfg = AppConfig()
        assert cfg.api_key == "a-real-random-secret-value"
        assert "Could not read API_KEY_FILE" in caplog.text

    def test_no_file_variant_is_fully_backward_compatible(self, monkeypatch):
        """Purely additive (P0-4): with no *_FILE vars set at all, behavior is
        byte-for-byte identical to before this feature existed."""
        monkeypatch.setenv("API_KEY", "a-real-random-secret-value")
        cfg = AppConfig()
        assert cfg.api_key == "a-real-random-secret-value"


class TestEnvOrFileHelper:
    """Direct tests of the _env_or_file() helper used by API_KEY, DATABASE_URL,
    S3_ACCESS_KEY, S3_SECRET_KEY, and SMTP_PASSWORD (P0-4)."""

    def test_plain_env_var_used_when_no_file_var_set(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET_FILE", raising=False)
        monkeypatch.setenv("MY_SECRET", "plain-value")
        assert _env_or_file("MY_SECRET") == "plain-value"

    def test_default_used_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.delenv("MY_SECRET_FILE", raising=False)
        assert _env_or_file("MY_SECRET", "fallback") == "fallback"

    def test_file_content_is_stripped(self, monkeypatch, tmp_path):
        f = tmp_path / "secret"
        f.write_text("  value-with-whitespace  \n", encoding="utf-8")
        monkeypatch.setenv("MY_SECRET_FILE", str(f))
        assert _env_or_file("MY_SECRET") == "value-with-whitespace"


class TestSecretRedaction:
    """P0-4: redaction utilities for any future diagnostic/support-bundle
    tool, plus a regression guard that log_config_summary() never leaks an
    actual secret value into logs."""

    def test_is_secret_env_var_matches_known_patterns(self):
        from people_analytics_service.config import is_secret_env_var

        for name in ("API_KEY", "API_KEY_FILE", "POSTGRES_PASSWORD", "S3_SECRET_KEY", "SOME_TOKEN"):
            assert is_secret_env_var(name), name

    def test_is_secret_env_var_does_not_match_unrelated_names(self):
        from people_analytics_service.config import is_secret_env_var

        for name in ("SERVICE_PORT", "YOLO_IMGSZ", "LOG_FORMAT", "S3_ENDPOINT"):
            assert not is_secret_env_var(name), name

    def test_redact_env_dict_masks_only_secret_keys(self):
        from people_analytics_service.config import redact_env_dict

        env = {"API_KEY": "hunter2", "SERVICE_PORT": "18000", "MINIO_ROOT_PASSWORD": "sw0rdfish"}
        redacted = redact_env_dict(env)
        assert redacted["API_KEY"] == "***REDACTED***"
        assert redacted["MINIO_ROOT_PASSWORD"] == "***REDACTED***"
        assert redacted["SERVICE_PORT"] == "18000"

    def test_redact_env_dict_leaves_empty_values_alone(self):
        """An unset/empty secret is not itself sensitive — don't mask "" to
        avoid implying a value is set when it isn't."""
        from people_analytics_service.config import redact_env_dict

        assert redact_env_dict({"API_KEY": ""}) == {"API_KEY": ""}

    def test_redact_url_credentials_masks_password_only(self):
        from people_analytics_service.config import redact_url_credentials

        redacted = redact_url_credentials("postgresql://safeaging:hunter2@postgres:5432/safeaging")
        assert "hunter2" not in redacted
        assert redacted == "postgresql://safeaging:***REDACTED***@postgres:5432/safeaging"

    def test_log_config_summary_never_logs_actual_secret_values(self, monkeypatch, caplog):
        """Regression guard: however log_config_summary() evolves, it must
        never interpolate the raw DATABASE_URL (which embeds the Postgres
        password), S3_SECRET_KEY, or SMTP_PASSWORD into a log line."""
        import logging as _logging

        from people_analytics_service import config as config_module

        secret_db_url = "postgresql://safeaging:totally-secret-db-password@postgres:5432/safeaging"
        monkeypatch.setattr(config_module, "DATABASE_URL", secret_db_url)
        monkeypatch.setattr(config_module, "S3_SECRET_KEY", "totally-secret-s3-key-value")
        monkeypatch.setattr(config_module, "S3_ENDPOINT", "http://minio:9000")
        monkeypatch.setattr(config_module, "SMTP_PASSWORD", "totally-secret-smtp-password")

        with caplog.at_level(_logging.INFO):
            config_module.log_config_summary()

        log_text = caplog.text
        assert "totally-secret-db-password" not in log_text
        assert secret_db_url not in log_text
        assert "totally-secret-s3-key-value" not in log_text
        assert "totally-secret-smtp-password" not in log_text
