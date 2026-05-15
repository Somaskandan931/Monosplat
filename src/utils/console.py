"""
console.py
Configure stdout/stderr for Windows consoles so Unicode log symbols do not crash startup.
"""

import sys
from pathlib import Path


def configure_console_encoding() -> None:
    """Use UTF-8 on Windows terminals that default to cp1252."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def ensure_project_dirs() -> None:
    """Create runtime directories expected by the README quick-start flow."""
    for rel in (
        "uploads",
        "work",
        "outputs/logs",
        "outputs/renders",
        "outputs/videos",
        "models/gaussian",
        "models/checkpoints",
    ):
        Path(rel).mkdir(parents=True, exist_ok=True)
