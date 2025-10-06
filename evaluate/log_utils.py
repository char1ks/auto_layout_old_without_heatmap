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
        rich_console = Console(file=sys.stderr, force_terminal=True, force_interactive=True, color_system="truecolor")
        rich_handler = RichHandler(rich_tracebacks=True, console=rich_console, show_path=False, show_level=False)
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        root_logger.setLevel(lvl)
        root_logger.addHandler(rich_handler)
    except Exception:
        logging.basicConfig(
            level=lvl,
            format="[%(levelname)s] %(name)s: %(message)s",
            force=True,
        )

def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    lvl_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    lvl_num = logging.getLevelNamesMapping().get(lvl_name, logging.INFO)
    logger.setLevel(lvl_num)
    return logger
