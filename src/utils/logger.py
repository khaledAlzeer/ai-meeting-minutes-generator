"""Application-wide logging configuration.

Centralizing logging setup here ensures every module gets consistently
formatted output and avoids duplicate handlers being attached when
modules are imported multiple times (e.g. by Gradio's reloader).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger exactly once per process.

    Args:
        level: Logging level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        log_file: Optional path to also write logs to a file, in addition
            to stdout.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    # Avoid duplicate handlers if this is somehow called more than once.
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Third-party libraries can be extremely verbose at INFO/DEBUG level;
    # keep them quieter unless the user explicitly wants DEBUG everywhere.
    if resolved_level > logging.DEBUG:
        for noisy_logger in ("urllib3", "httpx", "httpcore", "filelock"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging with defaults if needed."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
