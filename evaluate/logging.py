from __future__ import annotations

import logging
import os
from typing import Optional
from rich.logging import RichHandler

def setup_logging(level: Optional[str] = None) -> None:
    if logging.getLogger().handlers:
        return

    lvl_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    lvl = logging.getLevelNamesMapping().get(lvl_name, logging.INFO)

    try:
        logging.basicConfig(
            level=lvl,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except Exception:
        logging.basicConfig(level=lvl, format="[%(levelname)s] %(name)s: %(message)s")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or __name__)
