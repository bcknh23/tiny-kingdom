"""Logging configuration: console plus a rotating file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import DEFAULT_LOG_PATH

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO", log_path: Path | None = DEFAULT_LOG_PATH) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_paperbot", False):
            root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    console._paperbot = True  # type: ignore[attr-defined]
    root.addHandler(console)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        file_handler._paperbot = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
