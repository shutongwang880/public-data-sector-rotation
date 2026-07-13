"""Unified feature preprocessing and master dataset construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import get_path, load_config
from src.data.align import apply_feature_lags, fill_features, to_month_start
from src.data.catalog import (
    FEATURE_CATEGORIES,
    INDUSTRY_FACTOR_COLS,
    MARKET_STATE_COLS,
    ZSCORE_COLS,
    get_publication_lags,
    get_required_features,
    zscore_col,
)
from src.data.feature_selection import get_selected_global_raw
from src.data.industry_mask import (
    filter_training_sample,
    save_industry_metadata,
    truncate_late_start_industries,
)
from src.fetch.macro_data import compute_tsf_stock_yoy


def _repair_tsf_column(features: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    increments_path = raw_dir / "tsf_increments.csv"
    if not increments_path.exists() or "TSF" not in features.columns:
        return features
    increments = pd.read_csv(increments_path, parse_dates=["date"])
    tsf = compute_tsf_stock_yoy(increments)
    out = features.drop(columns=["TSF"], errors="ignore").merge(tsf, on="date", how="left")
    return out.sort_values("date")


def _load_feature_universe(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "feature_universe.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    macro = pd.read_csv(raw_dir / "macro_indicators.csv", parse_dates=["date"])
    liquidity = pd.read_csv(raw_dir / "feature_liquidity.csv", parse_dates=["date"])
    market_state = pd.read_csv(raw_dir / "feature_market_state.csv", parse_dates=["date"])
    industry = pd.read_csv(raw_dir / "feature_industry_factors.csv", parse_dates=["date"])

    universe = macro
    for frame in [liquidity, market_state, industry]:
        universe = universe.merge(frame, on="date", how="outer")
    return universe.sort_values("date")


def standardize_features(
    features: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = features.copy()
    valid_cols = [c for c in feature_cols if c in scaled.columns]
    scaled_values = scaler.fit_transform(scaled[valid_cols].values)
    for i, col in enumerate(valid_cols):
        scaled[zscore_col(col)] = scaled_values[:, i]
    return scaled, scaler


def merge_master_dataset(
    features_lagged: pd.DataFrame,
    industry_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> pd.DataFrame:
    merged = features_lagged.merge(industry_returns, on="date", how="inner")
    merged = merged.merge(benchmark_returns, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.insert(0, "Date", merged["date"].dt.strftime("%Y-%m"))
    return merged


def run_preprocessing(
    features: pd.DataFrame | None = None,
    industry_returns: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame | StandardScaler]:
    raw_dir = get_path("raw")
    processed_dir = get_path("processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    if features is None:
        features = _load_feature_universe(raw_dir)
    if industry_returns is None:
        industry_returns = pd.read_csv(raw_dir / "industry_monthly_returns.csv", parse_dates=["date"])
    if benchmark_returns is None:
        benchmark_returns = pd.read_csv(raw_dir / "hs300_monthly_returns.csv", parse_dates=["date"])

    industry_returns = truncate_late_start_industries(to_month_start(industry_returns))
    save_industry_metadata(industry_returns)

    features = to_month_start(features)
    benchmark_returns = to_month_start(benchmark_returns)
    features = _repair_tsf_column(features, raw_dir)

    required = get_required_features()
    selected = get_selected_global_raw()
    available = [c for c in selected if c in features.columns]

    # TSF, industry factors, market/liquidity extended: no backward fill before lag
    no_bfill = {
        "TSF",
        *INDUSTRY_FACTOR_COLS,
        *MARKET_STATE_COLS,
        "MARGIN_CHG",
        "TURNOVER_ZSCORE",
        "ETF_FLOW_MOM",
    }
    prelag_fillable = [c for c in available if c not in no_bfill]
    features_clean = fill_features(features, prelag_fillable)
    for col in no_bfill:
        if col in features.columns:
            features_clean[col] = features[col]

    features_lagged = apply_feature_lags(features_clean, available, get_publication_lags(), list(no_bfill))
    postlag_fillable = [c for c in available if c not in no_bfill]
    features_lagged[postlag_fillable] = fill_features(
        features_lagged[postlag_fillable], postlag_fillable, backfill=False
    )

    features_scaled, scaler = standardize_features(features_lagged, available)
    merged = merge_master_dataset(features_lagged, industry_returns, benchmark_returns)

    required_present = [c for c in required if c in merged.columns]
    merged_clean = merged.dropna(subset=required_present).reset_index(drop=True)
    merged_train = filter_training_sample(merged_clean)

    if save:
        features_clean.to_csv(processed_dir / "features_clean.csv", index=False, encoding="utf-8-sig")
        features_lagged.to_csv(processed_dir / "features_lagged.csv", index=False, encoding="utf-8-sig")
        features_scaled.to_csv(processed_dir / "features_standardized.csv", index=False, encoding="utf-8-sig")
        merged_clean.to_csv(processed_dir / "master_dataset.csv", index=False, encoding="utf-8-sig")
        merged_train.to_csv(processed_dir / "master_dataset_train.csv", index=False, encoding="utf-8-sig")

        # backward-compatible aliases
        macro_cols = [c for c in get_selected_global_raw()[:8] if c in features_clean.columns]
        features_clean[macro_cols + ["date"]].to_csv(
            processed_dir / "macro_clean.csv", index=False, encoding="utf-8-sig"
        )
        features_lagged[macro_cols + ["date"]].to_csv(
            processed_dir / "macro_lagged.csv", index=False, encoding="utf-8-sig"
        )
        features_scaled[[c for c in features_scaled.columns if c in get_selected_global_raw() or c.endswith("_zscore")]].to_csv(
            processed_dir / "macro_standardized.csv", index=False, encoding="utf-8-sig"
        )

        pd.DataFrame(
            [{"category": k, "features": ", ".join(v)} for k, v in FEATURE_CATEGORIES.items()]
        ).to_csv(processed_dir / "feature_catalog.csv", index=False, encoding="utf-8-sig")

    return {
        "features_clean": features_clean,
        "features_lagged": features_lagged,
        "features_scaled": features_scaled,
        "master": merged_train,
        "master_full": merged_clean,
        "scaler": scaler,
        "feature_cols": available,
    }
