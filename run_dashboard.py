#!/usr/bin/env python3
"""Launch the Streamlit Business Analytics dashboard."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_runtime_deps

ensure_runtime_deps("streamlit")
APP = PROJECT_ROOT / "src" / "dashboard" / "streamlit_app.py"


def _clear_pycache() -> None:
    for cache in (PROJECT_ROOT / "src").rglob("__pycache__"):
        if cache.is_dir():
            for p in cache.iterdir():
                p.unlink(missing_ok=True)
            cache.rmdir()


def _verify_imports() -> None:
    importlib.import_module("src.dashboard.data_loader")
    from src.dashboard.data_loader import load_live_forecast  # noqa: F401


def main() -> None:
    _clear_pycache()
    _verify_imports()
    print("Starting dashboard at http://localhost:8501")
    print("Press Ctrl+C to stop.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
            "--server.headless",
            "true",
            "--server.runOnSave",
            "false",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
