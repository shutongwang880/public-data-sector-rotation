"""Fetch Shenwan Level-1 industry index data."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

from src.config import get_path, load_config


def get_sw_industry_list() -> pd.DataFrame:
    """Load SW L1 list; fall back to local cache when legulegu.com is unreachable."""
    cache = get_path("raw") / "sw_industry_list.csv"
    try:
        info = ak.sw_index_first_info()
        info = info.rename(columns={"行业代码": "code", "行业名称": "name"})
        info["code"] = info["code"].str.replace(".SI", "", regex=False)
        return info[["code", "name"]]
    except Exception as exc:
        if cache.exists():
            print(f"  ⚠ 在线行业列表失败 ({exc.__class__.__name__})，使用本地缓存 {cache.name}")
            return pd.read_csv(cache)
        raise RuntimeError(
            "无法获取申万行业列表，且本地无 sw_industry_list.csv。"
            "请检查网络或稍后重试。"
        ) from exc


def fetch_industry_monthly_prices(code: str) -> pd.DataFrame:
    df = ak.index_hist_sw(symbol=code, period="month")
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date", "close"]].dropna()


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.sort_values("date").copy()
    out["return"] = out["close"].pct_change()
    return out.dropna(subset=["return"])


def fetch_all_industry_returns(save: bool = True, sleep_seconds: float = 0.3) -> pd.DataFrame:
    cfg = load_config()
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])

    industries = get_sw_industry_list()
    total = len(industries)
    return_frames: list[pd.DataFrame] = []

    print(f"  共 {total} 个申万一级行业，逐个从 swsresearch.com 拉取（每个约 5–30 秒，请耐心等待）")
    sys.stdout.flush()

    for i, row in industries.iterrows():
        code, name = row["code"], row["name"]
        idx = int(i) + 1 if isinstance(i, int) else len(return_frames) + 1
        print(f"  [{idx}/{total}] 正在拉取 {name} ({code}) ...", end="", flush=True)
        t0 = time.time()
        try:
            prices = fetch_industry_monthly_prices(code)
            rets = compute_monthly_returns(prices)
            rets = rets[(rets["date"] >= start) & (rets["date"] <= end)]
            rets = rets.rename(columns={"return": name})[["date", name]]
            return_frames.append(rets)
            elapsed = time.time() - t0
            print(f" ✓ {len(rets)} 个月 ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f" ✗ 失败 ({elapsed:.1f}s): {exc}")
        sys.stdout.flush()
        time.sleep(sleep_seconds)

    if not return_frames:
        raise RuntimeError("No industry return data fetched.")

    industry_returns = return_frames[0]
    for frame in return_frames[1:]:
        industry_returns = industry_returns.merge(frame, on="date", how="outer")

    industry_returns = industry_returns.sort_values("date")

    if save:
        raw_dir = get_path("raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        industries.to_csv(raw_dir / "sw_industry_list.csv", index=False, encoding="utf-8-sig")
        industry_returns.to_csv(raw_dir / "industry_monthly_returns.csv", index=False, encoding="utf-8-sig")

    print(f"  行业数据完成: {industry_returns.shape[1] - 1} 行业 × {industry_returns.shape[0]} 月")
    sys.stdout.flush()
    return industry_returns


if __name__ == "__main__":
    data = fetch_all_industry_returns()
    print(data.head())
    print(f"Industries: {data.shape[1] - 1}, months: {data.shape[0]}")
