"""Fetch benchmark index data (HS300, CSI500)."""

from __future__ import annotations

import akshare as ak
import pandas as pd

from src.config import get_path, load_config


def fetch_hs300_daily() -> pd.DataFrame:
    cfg = load_config()
    symbol = cfg["benchmark"]["symbol"]
    df = ak.stock_zh_index_daily(symbol=symbol)
    df = df.rename(columns={"date": "date", "close": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df[["date", "close", "volume"]].dropna(subset=["date", "close"]).sort_values("date")
    return df[["date", "close"]].dropna().sort_values("date")


def compute_monthly_returns(daily: pd.DataFrame, col_name: str = "HS300") -> pd.DataFrame:
    monthly = daily.set_index("date").resample("ME")["close"].last().dropna()
    returns = monthly.pct_change().dropna()
    out = returns.reset_index()
    out.columns = ["date", col_name]
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def fetch_index_daily(symbol: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date", "close"]].dropna().sort_values("date")


def fetch_csi500_returns(save: bool = True) -> pd.DataFrame:
    cfg = load_config()
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])
    symbol = cfg.get("benchmark", {}).get("csi500_symbol", "sh000905")

    raw_dir = get_path("raw")
    daily_path = raw_dir / "csi500_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path, parse_dates=["date"])
    else:
        daily = fetch_index_daily(symbol)
        if save:
            raw_dir.mkdir(parents=True, exist_ok=True)
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    returns = compute_monthly_returns(daily, "CSI500")
    returns = returns[(returns["date"] >= start) & (returns["date"] <= end)]
    if save:
        returns.to_csv(raw_dir / "csi500_monthly_returns.csv", index=False, encoding="utf-8-sig")
    return returns


def fetch_benchmark_returns(save: bool = True) -> pd.DataFrame:
    cfg = load_config()
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])

    daily = fetch_hs300_daily()
    returns = compute_monthly_returns(daily)
    returns = returns[(returns["date"] >= start) & (returns["date"] <= end)]

    if save:
        raw_dir = get_path("raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(raw_dir / "hs300_daily.csv", index=False, encoding="utf-8-sig")
        returns.to_csv(raw_dir / "hs300_monthly_returns.csv", index=False, encoding="utf-8-sig")

    return returns


if __name__ == "__main__":
    data = fetch_benchmark_returns()
    print(data.head())
    print(data.tail())
