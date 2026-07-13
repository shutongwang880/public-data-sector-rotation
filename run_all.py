#!/usr/bin/env python3
"""
Run the full pipeline end-to-end.

Usage:
    python run_all.py              # full pipeline with data fetch
    python run_all.py --skip-fetch # use cached raw data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_runtime_deps

ensure_runtime_deps("pandas")


def run(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / script), *(extra_args or [])]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main(skip_fetch: bool = False) -> None:
    print("=" * 60)
    print("Public-Data Sector Rotation Decision Support System")
    print("=" * 60)

    phase1_args = ["--skip-fetch"] if skip_fetch else []
    run("run_phase1.py", phase1_args)
    run("run_ml.py")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("  docs/REPORT.docx        — business report (Word)")
    print("  docs/REPORT.md          — research summary (Markdown)")
    print("  python run_dashboard.py — interactive dashboard")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    main(skip_fetch=args.skip_fetch)
