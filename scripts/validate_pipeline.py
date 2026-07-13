#!/usr/bin/env python3
"""Quick pipeline validation for CI and local smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_path  # noqa: E402


REQUIRED_AFTER_ML = [
    "ml/model_comparison_primary.csv",
    "ml/benchmark_comparison.csv",
    "ml/backtest_lambdarank.csv",
    "ml/predictions_lambdarank_nested.csv",
    "ml/analysis/ic_decay_summary.csv",
]

REQUIRED_AFTER_PHASE1 = [
    "master_dataset_train.csv",
]


def check_files(rel_paths: list[str], base_key: str = "processed") -> list[str]:
    base = get_path(base_key)
    missing = []
    for rel in rel_paths:
        if not (base / rel).exists():
            missing.append(str(base / rel))
    return missing


def main(strict_ml: bool = False) -> int:
    errors: list[str] = []

    # Import smoke test
    try:
        from src.dashboard.data_loader import load_live_forecast  # noqa: F401
        from src.ml.nested_walkforward import walk_forward_lambdarank_nested  # noqa: F401
    except Exception as e:
        errors.append(f"Import failed: {e}")

    phase1_missing = check_files(REQUIRED_AFTER_PHASE1)
    if phase1_missing:
        errors.append(f"Phase1 outputs missing ({len(phase1_missing)} files). Run: python run_phase1.py --skip-fetch")

    if strict_ml:
        ml_missing = check_files(REQUIRED_AFTER_ML)
        if ml_missing:
            errors.append(f"ML outputs missing ({len(ml_missing)} files). Run: python run_ml.py")

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strict-ml", action="store_true", help="Require full ML outputs")
    args = p.parse_args()
    raise SystemExit(main(strict_ml=args.strict_ml))
