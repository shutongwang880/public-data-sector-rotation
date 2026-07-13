"""Monthly alignment and publication lag utilities."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.catalog import ALL_FEATURE_COLS, get_publication_lags


def to_month_start(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[date_col] = out[date_col].dt.to_period("M").dt.to_timestamp()
    return out.sort_values(date_col).drop_duplicates(date_col, keep="last")


def daily_to_monthly_last(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.set_index(date_col).resample("ME").last()
    out.index = out.index.to_period("M").to_timestamp()
    return out.reset_index().rename(columns={"index": "date"})


def daily_to_monthly_sum(df: pd.DataFrame, value_cols: list[str], date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    monthly = out.set_index(date_col)[value_cols].resample("ME").sum()
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly.reset_index()


def daily_to_monthly_mean(df: pd.DataFrame, value_cols: list[str], date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    monthly = out.set_index(date_col)[value_cols].resample("ME").mean()
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly.reset_index()


def fill_features(
    df: pd.DataFrame,
    columns: list[str],
    backfill: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    valid = [c for c in columns if c in out.columns]
    out[valid] = out[valid].ffill()
    if backfill:
        out[valid] = out[valid].bfill()
    return out


def apply_feature_lags(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lags: dict[str, int] | None = None,
    no_backfill_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Shift features to reflect information availability at decision time."""
    lags = lags or get_publication_lags()
    columns = columns or [c for c in ALL_FEATURE_COLS if c in df.columns]
    no_backfill_cols = set(no_backfill_cols or [])

    out = df.copy()
    for col in columns:
        lag = int(lags.get(col, lags.get("_default", 1)))
        if lag:
            out[col] = out[col].shift(lag)

    fillable = [c for c in columns if c not in no_backfill_cols]
    out[fillable] = fill_features(out[fillable], fillable, backfill=False)
    return out


def filter_sample_range(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    cfg = load_config()
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out[(out[date_col] >= start) & (out[date_col] <= end)].sort_values(date_col)
