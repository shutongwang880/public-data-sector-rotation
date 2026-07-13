"""Market state layer: trend, breadth, style rotation, and risk appetite."""

from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data.align import filter_sample_range, to_month_start


def _monthly_returns_from_daily(daily: pd.DataFrame, name: str) -> pd.DataFrame:
    daily = daily.sort_values("date").copy()
    daily["date"] = pd.to_datetime(daily["date"])
    monthly = daily.set_index("date")["close"].resample("ME").last().dropna()
    rets = monthly.pct_change().dropna()
    out = rets.reset_index()
    out.columns = ["date", name]
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def fetch_chinext_monthly_returns() -> pd.DataFrame:
    """ChiNext (创业板指) monthly returns."""
    cfg = load_config()
    symbol = cfg.get("market_state", {}).get("chinext_symbol", "sz399006")
    raw_path = get_path("raw") / "chinext_daily.csv"

    if raw_path.exists():
        daily = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        daily = ak.stock_zh_index_daily(symbol=symbol)
        daily["date"] = pd.to_datetime(daily["date"])
        daily.to_csv(raw_path, index=False, encoding="utf-8-sig")

    return _monthly_returns_from_daily(daily, "CYB")


def _rolling_cum_return(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window, min_periods=window).apply(lambda x: x.prod() - 1, raw=True)


def compute_hs300_trends(hs300_monthly: pd.DataFrame) -> pd.DataFrame:
    out = hs300_monthly[["date"]].copy()
    rets = hs300_monthly["HS300"]
    out["HS300_TREND_3M"] = _rolling_cum_return(rets, 3)
    out["HS300_TREND_6M"] = _rolling_cum_return(rets, 6)
    return out


def compute_market_breadth(industry_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional advance ratio: share of industries with positive monthly return.
    Legulegu breadth API is unstable; industry-level breadth is a robust proxy.
    """
    from src.data.industry_mask import get_comparable_industry_columns

    df = to_month_start(industry_returns)
    industries = get_comparable_industry_columns(df, for_date=None)
    rets = df[industries]
    breadth = (rets > 0).sum(axis=1) / rets.notna().sum(axis=1)
    return pd.DataFrame({"date": df["date"], "MARKET_BREADTH": breadth})


def compute_cyb_vs_hs300(
    chinext_monthly: pd.DataFrame,
    hs300_monthly: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    merged = hs300_monthly.merge(chinext_monthly, on="date", how="inner").sort_values("date")
    cyb_cum = _rolling_cum_return(merged["CYB"], window)
    hs_cum = _rolling_cum_return(merged["HS300"], window)
    return pd.DataFrame(
        {
            "date": merged["date"],
            "CYB_VS_HS300": cyb_cum - hs_cum,
        }
    )


def compute_vol_trend(liquidity: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    vol = liquidity[["date", "HS300_VOL"]].sort_values("date").copy()
    vol["VOL_TREND"] = vol["HS300_VOL"] - vol["HS300_VOL"].shift(window)
    return vol[["date", "VOL_TREND"]]


def compute_hs300_ma_gaps(hs300_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Month-end (Price - MA) / MA gaps for HS300 trend regime."""
    raw_path = get_path("raw") / "hs300_daily.csv"
    if hs300_daily is None:
        if raw_path.exists():
            daily = pd.read_csv(raw_path, parse_dates=["date"])
        else:
            daily = ak.stock_zh_index_daily(symbol=load_config()["benchmark"]["symbol"])
            daily["date"] = pd.to_datetime(daily["date"])
    else:
        daily = hs300_daily.copy()

    daily = daily.sort_values("date")
    daily["ma20"] = daily["close"].rolling(20, min_periods=10).mean()
    daily["ma60"] = daily["close"].rolling(60, min_periods=30).mean()
    daily["HS300_MA20_GAP"] = (daily["close"] - daily["ma20"]) / daily["ma20"]
    daily["HS300_MA60_GAP"] = (daily["close"] - daily["ma60"]) / daily["ma60"]

    monthly = daily.set_index("date").resample("ME").last().reset_index()
    monthly["date"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    return monthly[["date", "HS300_MA20_GAP", "HS300_MA60_GAP"]]


def compute_realized_vol_features(hs300_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    raw_path = get_path("raw") / "hs300_daily.csv"
    if hs300_daily is None:
        daily = pd.read_csv(raw_path, parse_dates=["date"]) if raw_path.exists() else None
    else:
        daily = hs300_daily.copy()

    if daily is None or daily.empty:
        return pd.DataFrame(columns=["date", "HS300_VOL_20D", "HS300_VOL_60D", "HS300_VOL_CHG"])

    daily = daily.sort_values("date")
    daily["ret"] = daily["close"].pct_change()
    daily["HS300_VOL_20D"] = daily["ret"].rolling(20, min_periods=10).std() * np.sqrt(252)
    daily["HS300_VOL_60D"] = daily["ret"].rolling(60, min_periods=30).std() * np.sqrt(252)

    monthly = daily.set_index("date").resample("ME").last().reset_index()
    monthly["HS300_VOL_CHG"] = monthly["HS300_VOL_60D"] - monthly["HS300_VOL_60D"].shift(3)
    monthly["date"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    return monthly[["date", "HS300_VOL_20D", "HS300_VOL_60D", "HS300_VOL_CHG"]]


def fetch_all_market_state(
    industry_returns: pd.DataFrame | None = None,
    liquidity: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    raw_dir = get_path("raw")

    if industry_returns is None:
        industry_returns = pd.read_csv(raw_dir / "industry_monthly_returns.csv", parse_dates=["date"])
    if liquidity is None:
        liquidity = pd.read_csv(raw_dir / "feature_liquidity.csv", parse_dates=["date"])

    hs300_monthly = pd.read_csv(raw_dir / "hs300_monthly_returns.csv", parse_dates=["date"])
    chinext_monthly = fetch_chinext_monthly_returns()

    hs300_daily = pd.read_csv(raw_dir / "hs300_daily.csv", parse_dates=["date"]) if (
        raw_dir / "hs300_daily.csv"
    ).exists() else None

    frames = [
        compute_hs300_trends(hs300_monthly),
        compute_market_breadth(industry_returns),
        compute_cyb_vs_hs300(chinext_monthly, hs300_monthly),
        compute_vol_trend(liquidity),
        compute_hs300_ma_gaps(hs300_daily),
        compute_realized_vol_features(hs300_daily),
    ]

    state = frames[0]
    for frame in frames[1:]:
        state = state.merge(frame, on="date", how="outer")
    state = filter_sample_range(state.sort_values("date"))

    if save:
        raw_dir.mkdir(parents=True, exist_ok=True)
        state.to_csv(raw_dir / "feature_market_state.csv", index=False, encoding="utf-8-sig")

    return state
