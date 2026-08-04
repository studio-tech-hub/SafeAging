"""Unit tests for tools/check_compose_security.py (P0-2, P0-3).

Run with:
    pytest tools/test_check_compose_security.py -v

These exercise the line-level detection logic directly (no real compose
files touched), including a regression test for a real bug caught during
P0-3 development: naive regexes matching "API_KEY" inside the longer
"API_KEY_REQUIRED".
"""
from __future__ import annotations

from pathlib import Path

import check_compose_security as lint


class TestApiKeyRequiredCheck:
    def test_hardcoded_false_is_flagged(self):
        violations: list[str] = []
        lint._check_api_key_required("- API_KEY_REQUIRED=false", Path("x.yml"), 1, violations)
        assert len(violations) == 1
        assert "hardcoded to 'false'" in violations[0]

    def test_weak_fallback_default_is_flagged(self):
        violations: list[str] = []
        lint._check_api_key_required('- "API_KEY_REQUIRED=${API_KEY_REQUIRED:-false}"', Path("x.yml"), 1, violations)
        assert len(violations) == 1

    def test_true_default_is_clean(self):
        violations: list[str] = []
        lint._check_api_key_required('- "API_KEY_REQUIRED=${API_KEY_REQUIRED:-true}"', Path("x.yml"), 1, violations)
        assert violations == []

    def test_does_not_false_positive_on_unrelated_line(self):
        violations: list[str] = []
        lint._check_api_key_required("- SERVICE_HOST=0.0.0.0", Path("x.yml"), 1, violations)
        assert violations == []


class TestHardcodedLiteralDetection:
    def test_list_style_hardcoded_literal_is_flagged(self):
        violations: list[str] = []
        lint._check_hardcoded_literal("API_KEY", "- API_KEY=hunter2000", Path("x.yml"), 1, violations)
        assert len(violations) == 1
        assert "hardcoded literal" in violations[0]

    def test_mapping_style_hardcoded_literal_is_flagged(self):
        violations: list[str] = []
        lint._check_hardcoded_literal("POSTGRES_PASSWORD", "POSTGRES_PASSWORD: hunter2000", Path("x.yml"), 1, violations)
        assert len(violations) == 1

    def test_interpolated_value_is_not_a_hardcoded_literal(self):
        violations: list[str] = []
        lint._check_hardcoded_literal(
            "POSTGRES_PASSWORD",
            'POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:?msg}"',
            Path("x.yml"),
            1,
            violations,
        )
        assert violations == []

    def test_does_not_match_a_differently_named_key(self):
        """GF_SECURITY_ADMIN_PASSWORD assigning a literal must not be reported
        as a hardcoded GRAFANA_ADMIN_PASSWORD — that's a different key."""
        violations: list[str] = []
        lint._check_hardcoded_literal("GRAFANA_ADMIN_PASSWORD", "GF_SECURITY_ADMIN_PASSWORD: admin", Path("x.yml"), 1, violations)
        assert violations == []


class TestInterpolationChecks:
    def test_required_syntax_is_clean(self):
        violations: list[str] = []
        lint._check_interpolations("API_KEY", '- "API_KEY=${API_KEY:?Set API_KEY in .env}"', Path("x.yml"), 1, violations)
        assert violations == []

    def test_weak_fallback_default_is_flagged(self):
        violations: list[str] = []
        lint._check_interpolations(
            "POSTGRES_PASSWORD",
            "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-safeaging_dev_password}",
            Path("x.yml"),
            1,
            violations,
        )
        assert len(violations) == 1
        assert "safeaging_dev_password" in violations[0]

    def test_strong_looking_fallback_default_is_not_flagged(self):
        """A `:-` fallback whose default isn't a known placeholder isn't our
        concern here — check_compose_security only catches *known* weak
        values, it can't judge arbitrary string strength."""
        violations: list[str] = []
        lint._check_interpolations(
            "POSTGRES_PASSWORD",
            "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-Xk92jf0aQ==}",
            Path("x.yml"),
            1,
            violations,
        )
        assert violations == []

    def test_bare_reference_with_no_guard_is_flagged(self):
        violations: list[str] = []
        lint._check_interpolations("API_KEY", "- API_KEY=${API_KEY}", Path("x.yml"), 1, violations)
        assert len(violations) == 1
        assert "no required-value guard" in violations[0]

    def test_embedded_in_larger_string_is_detected(self):
        """Regression: a var referenced inside a composite value (e.g.
        DATABASE_URL) must still be checked, not just when it's the direct
        assignment target."""
        violations: list[str] = []
        line = '- "DATABASE_URL=postgresql://user:${POSTGRES_PASSWORD:-safeaging_dev_password}@postgres:5432/db"'
        lint._check_interpolations("POSTGRES_PASSWORD", line, Path("x.yml"), 1, violations)
        assert len(violations) == 1

    def test_assigned_to_a_differently_named_key_is_still_detected(self):
        """Regression: GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}
        must be caught even though the YAML key isn't GRAFANA_ADMIN_PASSWORD."""
        violations: list[str] = []
        line = "GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}"
        lint._check_interpolations("GRAFANA_ADMIN_PASSWORD", line, Path("x.yml"), 1, violations)
        assert len(violations) == 1
        assert "admin" in violations[0]

    def test_does_not_false_positive_on_longer_variable_name(self):
        """Regression: the exact bug found during P0-3 dev — a naive regex for
        "API_KEY" must not match inside "API_KEY_REQUIRED"."""
        violations: list[str] = []
        line = '- "API_KEY_REQUIRED=${API_KEY_REQUIRED:-true}"'
        lint._check_interpolations("API_KEY", line, Path("x.yml"), 1, violations)
        assert violations == []


class TestCheckFileIntegration:
    def test_clean_file_has_no_violations(self, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text(
            'services:\n'
            '  analytics:\n'
            '    environment:\n'
            '      - "API_KEY_REQUIRED=${API_KEY_REQUIRED:-true}"\n'
            '      - "API_KEY=${API_KEY:?Set API_KEY in .env}"\n'
            '  postgres:\n'
            '    environment:\n'
            '      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}"\n',
            encoding="utf-8",
        )
        original_root = lint.REPO_ROOT
        try:
            lint.REPO_ROOT = tmp_path
            assert lint._check_file(compose) == []
        finally:
            lint.REPO_ROOT = original_root

    def test_insecure_file_reports_all_violations(self, tmp_path):
        compose = tmp_path / "docker-compose.yml"
        compose.write_text(
            "services:\n"
            "  analytics:\n"
            "    environment:\n"
            "      - API_KEY_REQUIRED=false\n"
            "      - API_KEY=hunter2000\n"
            "  postgres:\n"
            "    environment:\n"
            "      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-safeaging_dev_password}\n",
            encoding="utf-8",
        )
        original_root = lint.REPO_ROOT
        try:
            lint.REPO_ROOT = tmp_path
            violations = lint._check_file(compose)
        finally:
            lint.REPO_ROOT = original_root
        assert len(violations) == 3
