"""Industry comparability: truncation, exclusion, and per-month masks."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.data.industries import get_industry_columns


def get_industry_config() -> dict:
    return load_config().get("industry", {})


def get_excluded_industries() -> list[str]:
    return list(get_industry_config().get("exclude_from_training", []))


def get_industry_columns_from_df(df: pd.DataFrame) -> list[str]:
    return get_industry_columns(df)


def first_valid_dates(industry_returns: pd.DataFrame) -> pd.Series:
    industries = get_industry_columns_from_df(industry_returns)
    first_dates: dict[str, pd.Timestamp] = {}
    for col in industries:
        valid = industry_returns.loc[industry_returns[col].notna(), "date"]
        first_dates[col] = valid.min() if not valid.empty else pd.NaT
    return pd.Series(first_dates, name="first_valid_date")


def truncate_late_start_industries(industry_returns: pd.DataFrame) -> pd.DataFrame:
    """Mask observations before each industry's first valid month (structural missing)."""
    cfg = get_industry_config()
    if not cfg.get("truncate_late_start", True):
        return industry_returns

    out = industry_returns.copy()
    industries = get_industry_columns_from_df(out)
    first = first_valid_dates(out)

    for col in industries:
        start = first.get(col)
        if pd.isna(start):
            out[col] = float("nan")
            continue
        out.loc[out["date"] < start, col] = float("nan")

    return out


def get_comparable_industry_columns(
    df: pd.DataFrame,
    for_date: pd.Timestamp | None = None,
    min_obs_ratio: float | None = None,
) -> list[str]:
    """
    Industries usable at `for_date`: not excluded, and sufficiently observed in sample.
    """
    cfg = get_industry_config()
    excluded = set(get_excluded_industries())
    min_ratio = min_obs_ratio or cfg.get("comparable_min_obs_ratio", 0.85)
    training_start = pd.Timestamp(cfg.get("training_start_date", "2016-01-01"))

    industries = [c for c in get_industry_columns_from_df(df) if c not in excluded]
    if for_date is not None:
        valid = []
        for col in industries:
            hist = df.loc[df["date"] <= for_date, col]
            if hist.notna().any():
                valid.append(col)
        return valid

    window = df[df["date"] >= training_start]
    n = max(len(window), 1)
    comparable = []
    for col in industries:
        obs_ratio = window[col].notna().mean()
        if obs_ratio >= min_ratio:
            comparable.append(col)
    return comparable


def filter_training_sample(df: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    cfg = get_industry_config()
    start = pd.Timestamp(start_date or cfg.get("training_start_date", "2016-01-01"))
    out = df[df["date"] >= start].copy()
    return out.sort_values("date").reset_index(drop=True)


def build_industry_metadata(industry_returns: pd.DataFrame) -> pd.DataFrame:
    cfg = get_industry_config()
    first = first_valid_dates(industry_returns)
    rows = []
    for industry, start in first.items():
        rows.append(
            {
                "industry": industry,
                "first_valid_date": start,
                "excluded": industry in get_excluded_industries(),
                "comparable_from": cfg.get("training_start_date"),
            }
        )
    return pd.DataFrame(rows)


def save_industry_metadata(industry_returns: pd.DataFrame) -> pd.DataFrame:
    meta = build_industry_metadata(industry_returns)
    path = get_path("processed") / "industry_metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(path, index=False, encoding="utf-8-sig")
    return meta
