# =============================================================================
# Summary : Shared logging setup for all three PARORA entry points
#           (app.py, server.py, app_lite.py). Writes a rotating file under
#           logs/ plus stdout (so `docker logs` / a terminal running run.sh
#           shows it live), with one line per event: timestamp, level,
#           component, message.
#
#           Idempotent by design -- safe to call setup_logging() more than
#           once in the same process without duplicating handlers or log
#           lines. This matters specifically for Streamlit (app.py,
#           app_lite.py): Streamlit re-executes the whole script top to
#           bottom on every chat message, so a naive addHandler() call at
#           module level would attach a new file handler -- and duplicate
#           every subsequent log line -- on every single turn.
#
#           PARORA_LOG_LEVEL (env var) controls verbosity; default INFO.
#           PARORA_LOG_DIR (env var) overrides where the file lands; default
#           "<this file's directory>/logs".
# =============================================================================

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(os.getenv("PARORA_LOG_DIR", str(Path(__file__).parent / "logs")))
_LOG_FILE = _LOG_DIR / "parora.log"
_LEVEL = getattr(logging, os.getenv("PARORA_LOG_LEVEL", "INFO").upper(), logging.INFO)
_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(component: str) -> logging.Logger:
    """
    Configure (once) and return the logger for one entry point.

    Args:
        component: Short name for this process -- "app", "server", or
                   "app_lite" -- used as the logger name and shown in every
                   line so a shared log file (or shared `docker logs`
                   stream) can be told apart by source.

    Returns:
        A logger under the "parora" namespace, safe to call repeatedly.
    """
    root = logging.getLogger("parora")
    root.setLevel(_LEVEL)

    if not root.handlers:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

        file_handler = RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        # This module's own logger, not the component's -- announced once,
        # from whichever entry point happens to initialize logging first.
        root.propagate = False
        logging.getLogger("parora.logging_setup").info(
            "Logging initialized -> %s (level=%s)", _LOG_FILE, logging.getLevelName(_LEVEL)
        )

    return logging.getLogger(f"parora.{component}")
