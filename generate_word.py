#!/usr/bin/env python3
"""Generate Word research report (.docx)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.report.word_export import generate_word


def main() -> None:
    out = generate_word()
    print(f"Word report generated: {out}")


if __name__ == "__main__":
    main()
