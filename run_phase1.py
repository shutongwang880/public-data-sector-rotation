#!/usr/bin/env python3
"""
Phase 1 entry point: four-category feature acquisition, preprocessing, and EDA.

Usage:
    python run_phase1.py              # full pipeline
    python run_phase1.py --skip-fetch # use cached raw data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_runtime_deps

ensure_runtime_deps("pandas")

import pandas as pd

from src.config import get_path

from src.eda.analysis import run_eda
from src.fetch.benchmark_data import fetch_benchmark_returns
from src.fetch.feature_data import fetch_all_features
from src.fetch.industry_data import fetch_all_industry_returns
from src.preprocess.pipeline import run_preprocessing
from src.fetch.industry_specific import save_industry_specific_features


def main(skip_fetch: bool = False) -> None:
    print("=" * 60)
    print("Phase 1: Multi-Feature Data Layer")
    print("Public-Data Sector Rotation — Phase 1: Data Pipeline")
    print("=" * 60)

    if not skip_fetch:
        print("\n[1/5] Fetching Shenwan L1 industry data...")
        industry = fetch_all_industry_returns()
        print(f"  Industry returns: {industry.shape[0]} months × {industry.shape[1] - 1} industries")

        print("\n[2/5] Fetching HS300 benchmark...")
        benchmark = fetch_benchmark_returns()
        print(f"  HS300 returns: {benchmark.shape[0]} months")

        from src.fetch.benchmark_data import fetch_csi500_returns

        print("\n[2b/5] Fetching CSI500 benchmark...")
        csi500 = fetch_csi500_returns()
        print(f"  CSI500 returns: {csi500.shape[0]} months")

        print("\n[3/5] Fetching four-category feature universe...")
        features = fetch_all_features(industry_returns=industry)
        universe = features["universe"]
        print(f"  Feature universe: {universe.shape[0]} months × {universe.shape[1] - 1} features")
        for _, row in features["categories"].iterrows():
            print(f"    - {row['category']}: {row['features']}")
    else:
        print("\n[1-3/5] Skipping fetch, using cached raw data...")

    print("\n[4/5] Preprocessing (align, lag, merge)...")
    from src.fetch.feature_data import refresh_global_features

    refresh_global_features(save=True)
    result = run_preprocessing()
    master = result["master"]

    print("\n[4b] Industry features (proxies + fundamentals + drivers)...")
    raw_dir = get_path("raw")
    industry = pd.read_csv(raw_dir / "industry_monthly_returns.csv", parse_dates=["date"])
    hs300 = pd.read_csv(raw_dir / "hs300_monthly_returns.csv", parse_dates=["date"])
    ind_spec = save_industry_specific_features(industry, hs300)
    print(f"  Industry feature rows: {len(ind_spec)}")
    if not ind_spec.empty:
        fund_cols = [c for c in ind_spec.columns if c.startswith("IND_PE") or c.startswith("IND_PB")]
        drv_cols = [c for c in ind_spec.columns if c.startswith("IND_DRV_")]
        if fund_cols:
            filled = ind_spec[fund_cols[0]].notna().sum()
            print(f"  Fundamentals filled: {filled}/{len(ind_spec)} (need TUSHARE_TOKEN or manual CSV)")
        if drv_cols:
            filled = ind_spec[drv_cols[0]].notna().sum()
            print(f"  Driver variables filled: {filled}/{len(ind_spec)}")
    print(f"  Master dataset (full): {result['master_full'].shape[0]} rows")
    print(f"  Master dataset (train): {master.shape[0]} rows × {master.shape[1]} columns")
    print(f"  Feature columns: {len(result['feature_cols'])}")

    print("\n[5/5] Exploratory analysis...")
    eda = run_eda()
    print(f"  Descriptive stats saved for {len(eda['statistics'])} variables")

    print("\n" + "=" * 60)
    print("Phase 1 complete. Deliverables:")
    print("  data/raw/feature_*.csv     - four-category raw features")
    print("  data/processed/master_dataset.csv       - full sample")
    print("  data/processed/master_dataset_train.csv - comparable sample (2020+)")
    print("  data/figures/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 data pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data download")
    args = parser.parse_args()
    main(skip_fetch=args.skip_fetch)
