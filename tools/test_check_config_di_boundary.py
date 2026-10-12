"""Unit tests for tools/check_config_di_boundary.py (P2-3).

Run with:
    pytest tools/test_check_config_di_boundary.py -v
"""
from __future__ import annotations

from pathlib import Path

import check_config_di_boundary as lint


class TestConfigGlobalMirrors:
    def test_extracts_global_mirrors_from_real_config_py(self):
        mirrors = lint._config_global_mirrors(lint.CONFIG_PY)
        # A representative sample of real AppConfig-mirrored globals.
        assert "CONFIDENCE_THRESHOLD" in mirrors
        assert "ENABLE_ROI" in mirrors
        assert "TRACK_MIN_HITS" in mirrors
        assert "EDGE_SQLITE_PATH" in mirrors

    def test_non_mirror_names_are_excluded(self):
        mirrors = lint._config_global_mirrors(lint.CONFIG_PY)
        # logger, get_config, AppConfig, CONFIG_SCHEMA_VERSION, and
        # normalize_camera_id are not `NAME = CONFIG.attr` mirrors.
        assert "logger" not in mirrors
        assert "get_config" not in mirrors
        assert "AppConfig" not in mirrors
        assert "CONFIG" not in mirrors
        assert "CONFIG_SCHEMA_VERSION" not in mirrors
        assert "normalize_camera_id" not in mirrors


class TestCheckFile:
    def _write(self, tmp_path: Path, content: str) -> Path:
        f = tmp_path / "sample.py"
        f.write_text(content, encoding="utf-8")
        return f

    def test_flags_relative_global_import(self, tmp_path):
        mirrors = {"CONFIDENCE_THRESHOLD"}
        f = self._write(tmp_path, "from ...config import CONFIDENCE_THRESHOLD\n")
        violations = lint._check_file(f, mirrors)
        assert len(violations) == 1
        assert "CONFIDENCE_THRESHOLD" in violations[0]

    def test_flags_absolute_global_import(self, tmp_path):
        mirrors = {"ENABLE_ROI"}
        f = self._write(
            tmp_path,
            "from people_analytics_service.config import ENABLE_ROI\n",
        )
        violations = lint._check_file(f, mirrors)
        assert len(violations) == 1

    def test_allows_logger_and_get_config(self, tmp_path):
        mirrors = {"CONFIDENCE_THRESHOLD"}
        f = self._write(
            tmp_path,
            "from ...config import logger, get_config, AppConfig\n",
        )
        violations = lint._check_file(f, mirrors)
        assert violations == []

    def test_allows_unrelated_imports(self, tmp_path):
        mirrors = {"CONFIDENCE_THRESHOLD"}
        f = self._write(tmp_path, "from ...tracking import iou\n")
        violations = lint._check_file(f, mirrors)
        assert violations == []

    def test_multiple_violations_in_one_import_statement(self, tmp_path):
        mirrors = {"CONFIDENCE_THRESHOLD", "IOU_THRESHOLD"}
        f = self._write(
            tmp_path,
            "from ...config import (\n    CONFIDENCE_THRESHOLD,\n    IOU_THRESHOLD,\n    logger,\n)\n",
        )
        violations = lint._check_file(f, mirrors)
        assert len(violations) == 2


class TestRealGuardedFiles:
    def test_pipeline_and_repository_are_currently_clean(self):
        """Regression guard: the actual P2-1/P2-2 code this task migrated
        must pass the lint it introduces."""
        mirrors = lint._config_global_mirrors(lint.CONFIG_PY)
        all_violations: list[str] = []
        for f in lint._guarded_files():
            all_violations.extend(lint._check_file(f, mirrors))
        assert all_violations == [], "\n".join(all_violations)
