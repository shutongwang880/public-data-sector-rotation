"""Compute industry momentum and reversal factors from monthly returns."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.data.align import filter_sample_range, to_month_start
from src.data.industries import get_industry_columns


def compute_industry_factors(industry_returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build market-level industry style factors:
    - IND_MOM_1M/3M/6M: equal-weight industry momentum
    - IND_REV_1M: short-term reversal (negative prior month return)
    - IND_DISP_1M: cross-sectional return dispersion
    """
    if industry_returns is None:
        path = get_path("raw") / "industry_monthly_returns.csv"
        industry_returns = pd.read_csv(path, parse_dates=["date"])

    df = to_month_start(industry_returns)
    industries = get_industry_columns(df)
    rets = df[industries]

    ew = rets.mean(axis=1, skipna=True)
    comp_3 = (1 + rets).rolling(3, min_periods=2).apply(lambda s: s.prod() - 1, raw=True)
    comp_6 = (1 + rets).rolling(6, min_periods=3).apply(lambda s: s.prod() - 1, raw=True)
    comp_12 = (1 + rets).rolling(12, min_periods=6).apply(lambda s: s.prod() - 1, raw=True)

    out = pd.DataFrame({"date": df["date"]})
    out["IND_MOM_1M"] = ew
    out["IND_MOM_3M"] = comp_3.mean(axis=1)
    out["IND_MOM_6M"] = comp_6.mean(axis=1)
    out["IND_MOM_12M"] = comp_12.mean(axis=1)
    out["IND_REV_1M"] = -ew.shift(1)
    out["IND_DISP_1M"] = rets.std(axis=1, skipna=True)

    return out.dropna(subset=["IND_MOM_1M"]).reset_index(drop=True)


def fetch_all_industry_factors(
    industry_returns: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    factors = compute_industry_factors(industry_returns)
    factors = filter_sample_range(factors)

    if save:
        raw_dir = get_path("raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        factors.to_csv(raw_dir / "feature_industry_factors.csv", index=False, encoding="utf-8-sig")

    return factors
