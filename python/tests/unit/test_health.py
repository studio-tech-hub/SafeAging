"""Unit tests for health.compute_status() reason codes (P0-2: auth_disabled)."""
from __future__ import annotations

from people_analytics_service.health import compute_status


class TestComputeStatusAuthReasonCode:
    def test_not_ready_when_model_not_ok_ignores_auth_flag(self):
        status, reasons = compute_status(model_ok=False, deps={}, auth_disabled=True)
        assert status == "not_ready"
        assert reasons == []

    def test_healthy_when_auth_enabled_and_deps_up(self):
        status, reasons = compute_status(
            model_ok=True,
            deps={"postgres": "up", "object_storage": "up"},
            auth_disabled=False,
        )
        assert status == "healthy"
        assert "auth_disabled" not in reasons

    def test_degraded_with_auth_disabled_reason_code(self):
        status, reasons = compute_status(
            model_ok=True,
            deps={"postgres": "up", "object_storage": "up"},
            auth_disabled=True,
        )
        assert status == "degraded"
        assert "auth_disabled" in reasons

    def test_auth_disabled_combines_with_other_reason_codes(self):
        status, reasons = compute_status(
            model_ok=True,
            deps={"postgres": "down", "object_storage": "up"},
            auth_disabled=True,
        )
        assert status == "degraded"
        assert "auth_disabled" in reasons
        assert "db_unreachable" in reasons

    def test_auth_disabled_defaults_to_false(self):
        """Backward-compatible default: existing callers that don't pass
        auth_disabled must not suddenly get the new reason code."""
        status, reasons = compute_status(
            model_ok=True,
            deps={"postgres": "up", "object_storage": "up"},
        )
        assert status == "healthy"
        assert reasons == []
