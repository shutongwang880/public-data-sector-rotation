"""Per-industry price and risk signals computed from public return data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_path
from src.data.align import filter_sample_range, to_month_start
from src.data.industry_mask import get_comparable_industry_columns

INDUSTRY_PRICE_COLS = [
    "IND_PX_MOM_1M",
    "IND_PX_MOM_3M",
    "IND_PX_MOM_6M",
    "IND_PX_MOM_12M",
    "IND_REL_STRENGTH",
    "IND_VOL",
    "IND_MAX_DD",
    "IND_POS_MONTH_RATIO",
]


def _rolling_cum(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window, min_periods=max(2, window // 2)).apply(
        lambda x: x.prod() - 1, raw=True
    )


def _rolling_max_drawdown(rets: pd.Series, window: int = 6) -> pd.Series:
    def _max_dd(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        wealth = np.cumprod(1 + x)
        peak = np.maximum.accumulate(wealth)
        dd = wealth / peak - 1
        return float(dd.min())

    return rets.rolling(window, min_periods=2).apply(_max_dd, raw=True)


def compute_industry_price_signals(
    industry_returns: pd.DataFrame,
    hs300_col: str = "HS300",
) -> pd.DataFrame:
    """Build panel-level industry price / risk factors (no proprietary fundamentals)."""
    df = to_month_start(industry_returns)
    industries = get_comparable_industry_columns(df)
    hs300 = df[hs300_col] if hs300_col in df.columns else pd.Series(0.0, index=df.index)

    rows: list[dict] = []
    for industry in industries:
        rets = df[industry]
        mom_1 = rets
        mom_3 = _rolling_cum(rets, 3)
        mom_6 = _rolling_cum(rets, 6)
        mom_12 = _rolling_cum(rets, 12)
        hs300_1 = hs300
        vol = rets.rolling(3, min_periods=2).std() * np.sqrt(12)
        max_dd = _rolling_max_drawdown(rets, window=6)
        pos_ratio = (rets > 0).rolling(6, min_periods=3).mean()

        for i, row in df.iterrows():
            if pd.isna(rets.loc[i]):
                continue
            rows.append(
                {
                    "date": row["date"],
                    "industry": industry,
                    "IND_PX_MOM_1M": mom_1.loc[i],
                    "IND_PX_MOM_3M": mom_3.loc[i],
                    "IND_PX_MOM_6M": mom_6.loc[i],
                    "IND_PX_MOM_12M": mom_12.loc[i],
                    "IND_REL_STRENGTH": mom_1.loc[i] - hs300_1.loc[i],
                    "IND_VOL": vol.loc[i],
                    "IND_MAX_DD": max_dd.loc[i],
                    "IND_POS_MONTH_RATIO": pos_ratio.loc[i],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.dropna(subset=["IND_PX_MOM_1M"]).reset_index(drop=True)


def save_industry_price_signals(
    industry_returns: pd.DataFrame | None = None,
    hs300_returns: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    raw_dir = get_path("raw")
    if industry_returns is None:
        industry_returns = pd.read_csv(raw_dir / "industry_monthly_returns.csv", parse_dates=["date"])
    merged = to_month_start(industry_returns)
    if hs300_returns is not None:
        hs300_returns = to_month_start(hs300_returns)
        merged = merged.merge(hs300_returns, on="date", how="left")
    elif "HS300" not in merged.columns:
        hs = pd.read_csv(raw_dir / "hs300_monthly_returns.csv", parse_dates=["date"])
        merged = merged.merge(to_month_start(hs), on="date", how="left")

    signals = compute_industry_price_signals(merged)
    signals = filter_sample_range(signals)
    if save and not signals.empty:
        raw_dir.mkdir(parents=True, exist_ok=True)
        signals.to_csv(raw_dir / "feature_industry_price.csv", index=False, encoding="utf-8-sig")
    return signals
