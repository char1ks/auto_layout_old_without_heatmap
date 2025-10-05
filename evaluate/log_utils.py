from __future__ import annotations

import logging
import os
from typing import Optional
import sys
from rich.logging import RichHandler
from rich.console import Console 
def setup_logging(level: Optional[str] = None) -> None:
    lvl_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    lvl = logging.getLevelNamesMapping().get(lvl_name, logging.INFO)

    try:
        rich_handler = RichHandler(rich_tracebacks=True, console=Console(file=sys.stdout))
        logging.basicConfig(
            level=lvl,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[rich_handler],
            force=True,
        )
    except Exception:
        logging.basicConfig(
            level=lvl,
            format="[%(levelname)s] %(name)s: %(message)s",
            force=True,
        )

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or __name__)
