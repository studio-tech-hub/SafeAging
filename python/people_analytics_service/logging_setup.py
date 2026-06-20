"""Structured logging setup.

Toggle between plain-text (default) and JSON by setting LOG_FORMAT=json.
JSON format is intended for production and Docker log aggregation.
Call configure_logging() once at startup; safe to call multiple times.
"""
import json
import logging
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, ensure_ascii=False)


_TEXT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def configure_logging(log_format: str = "text", level: int = logging.INFO) -> None:
    """(Re-)configure the root logger formatter.

    Args:
        log_format: "json" for structured JSON output, any other value for text.
        level: root logger level (default INFO).
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter: logging.Formatter = (
        _JsonFormatter() if log_format == "json" else _TEXT_FORMATTER
    )

    if not root.handlers:
        root.addHandler(logging.StreamHandler())

    for handler in root.handlers:
        handler.setFormatter(formatter)
