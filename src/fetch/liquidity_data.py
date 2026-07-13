"""Fetch market liquidity and risk appetite features."""

from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data.align import daily_to_monthly_mean, daily_to_monthly_sum, filter_sample_range, to_month_start


def fetch_margin_balance_monthly() -> pd.DataFrame:
    """Monthly margin financing balance (亿元)."""
    df = ak.stock_margin_account_info()
    out = df.rename(columns={"日期": "date", "融资余额": "MARGIN_BAL"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    monthly = daily_to_monthly_mean(out, ["MARGIN_BAL"])
    return to_month_start(monthly)[["date", "MARGIN_BAL"]]


def fetch_northbound_flow_monthly() -> pd.DataFrame:
    """Monthly northbound net purchase amount (亿元)."""
    df = ak.stock_hsgt_hist_em(symbol="北向资金")
    out = df.rename(columns={"日期": "date", "当日成交净买额": "NORTH_FLOW"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    monthly = daily_to_monthly_sum(out, ["NORTH_FLOW"])
    return to_month_start(monthly)[["date", "NORTH_FLOW"]]


def fetch_market_turnover_vol_monthly(hs300_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly turnover (volume) and realized volatility from HS300 daily data."""
    if hs300_daily is None:
        raw_path = get_path("raw") / "hs300_daily.csv"
        if raw_path.exists():
            daily = pd.read_csv(raw_path, parse_dates=["date"])
        else:
            daily = ak.stock_zh_index_daily(symbol=load_config()["benchmark"]["symbol"])
            daily["date"] = pd.to_datetime(daily["date"])
    else:
        daily = hs300_daily.copy()

    if "volume" not in daily.columns:
        daily = ak.stock_zh_index_daily(symbol=load_config()["benchmark"]["symbol"])
        daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values("date")
    daily["ret"] = daily["close"].pct_change()

    turnover = daily_to_monthly_sum(daily.rename(columns={"volume": "TURNOVER"}), ["TURNOVER"])

    vol = daily.copy()
    vol["date"] = pd.to_datetime(vol["date"])
    vol = vol.set_index("date")
    monthly_vol = vol["ret"].resample("ME").std() * np.sqrt(252)
    monthly_vol = monthly_vol.reset_index()
    monthly_vol.columns = ["date", "HS300_VOL"]
    monthly_vol["date"] = monthly_vol["date"].dt.to_period("M").dt.to_timestamp()

    out = turnover.merge(monthly_vol, on="date", how="outer")
    return to_month_start(out)


def compute_turnover_zscore(liquidity: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    out = liquidity[["date", "TURNOVER"]].sort_values("date").copy()
    roll_mean = out["TURNOVER"].rolling(window, min_periods=12).mean()
    roll_std = out["TURNOVER"].rolling(window, min_periods=12).std()
    out["TURNOVER_ZSCORE"] = (out["TURNOVER"] - roll_mean) / roll_std.replace(0, np.nan)
    return out[["date", "TURNOVER_ZSCORE"]]


def compute_margin_change(liquidity: pd.DataFrame) -> pd.DataFrame:
    margin = liquidity[["date", "MARGIN_BAL"]].sort_values("date").copy()
    margin["MARGIN_CHG"] = margin["MARGIN_BAL"].pct_change()
    return margin[["date", "MARGIN_CHG"]]


def fetch_etf_flow_momentum() -> pd.DataFrame:
    """Sector ETF monthly return momentum as public liquidity/industry-flow proxy."""
    etf_symbols = ["512480", "512000", "512010"]
    frames: list[pd.DataFrame] = []
    for sym in etf_symbols:
        try:
            daily = ak.fund_etf_hist_em(symbol=sym, period="daily", adjust="")
            daily["date"] = pd.to_datetime(daily["日期"])
            daily = daily.sort_values("date")
            monthly = daily.set_index("date")["收盘"].resample("ME").last().pct_change()
            out = monthly.reset_index()
            out.columns = ["date", sym]
            out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
            frames.append(out)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "ETF_FLOW_MOM"])

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    ret_cols = [c for c in merged.columns if c != "date"]
    merged["ETF_FLOW_MOM"] = merged[ret_cols].mean(axis=1)
    return merged[["date", "ETF_FLOW_MOM"]]


def fetch_all_liquidity(save: bool = True) -> pd.DataFrame:
    margin = fetch_margin_balance_monthly()
    north = fetch_northbound_flow_monthly()
    market = fetch_market_turnover_vol_monthly()

    liquidity = margin.merge(north, on="date", how="outer").merge(market, on="date", how="outer")
    liquidity = liquidity.sort_values("date")
    liquidity = liquidity.merge(compute_turnover_zscore(liquidity), on="date", how="left")
    liquidity = liquidity.merge(compute_margin_change(liquidity), on="date", how="left")

    try:
        etf = fetch_etf_flow_momentum()
        if not etf.empty:
            liquidity = liquidity.merge(etf, on="date", how="left")
    except Exception:
        pass

    liquidity = filter_sample_range(liquidity)

    if save:
        raw_dir = get_path("raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        liquidity.to_csv(raw_dir / "feature_liquidity.csv", index=False, encoding="utf-8-sig")

    return liquidity
