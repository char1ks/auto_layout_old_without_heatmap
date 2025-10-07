import os
if "MPLBACKEND" not in os.environ or os.environ["MPLBACKEND"].startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"
from .log_utils import setup_logging

setup_logging()