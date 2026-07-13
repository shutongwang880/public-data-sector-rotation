"""Re-launch scripts with project venv when dependencies are missing."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def venv_python(root: Path | None = None) -> Path | None:
    root = root or project_root()
    for name in ("python3", "python"):
        candidate = root / ".venv" / "bin" / name
        if candidate.exists():
            return candidate
    return None


def ensure_runtime_deps(*modules: str, relaunch: bool = True) -> None:
    """Exit or re-exec with .venv when imports fail."""
    missing = []
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if not missing:
        return

    root = project_root()
    py = venv_python(root)
    if relaunch and py is not None:
        print(f"Missing modules: {', '.join(missing)}")
        print(f"Re-launching with {py} ...")
        os.execv(str(py), [str(py), *sys.argv])

    print("Python dependencies are not installed.")
    print("Run:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    raise SystemExit(1)
