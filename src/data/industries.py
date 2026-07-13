"""Industry column helpers for panels and feature engineering."""

from __future__ import annotations

import pandas as pd

from src.data.catalog import ALL_FEATURE_COLS, MARKET_STATE_COLS, ZSCORE_COLS


def get_industry_columns(df: pd.DataFrame) -> list[str]:
    """Return Shenwan industry return columns from a master dataset."""
    exclude = {
        "Date",
        "date",
        "HS300",
        *ALL_FEATURE_COLS,
        *ZSCORE_COLS,
        "growth_high",
        "inflation_high",
        "quadrant_id",
        "quadrant_label",
        "cluster_id",
        "cluster_label",
        "regime_method",
        "growth_threshold_used",
        "inflation_threshold_used",
        "rolling_cluster_id",
        "rolling_cluster_label",
        *MARKET_STATE_COLS,
    }
    return [c for c in df.columns if c not in exclude]
