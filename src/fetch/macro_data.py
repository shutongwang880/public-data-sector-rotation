"""Fetch macroeconomic indicators via AKShare."""

from __future__ import annotations

import re
from pathlib import Path

import akshare as ak
import pandas as pd

from src.config import get_path, load_config


def parse_chinese_month(text: str) -> pd.Timestamp:
    """Parse strings like '2026年05月份' or '201501'."""
    text = str(text).strip()
    match = re.match(r"(\d{4})年(\d{1,2})月份", text)
    if match:
        year, month = match.groups()
        return pd.Timestamp(year=int(year), month=int(month), day=1)

    match = re.match(r"^(\d{4})(\d{2})$", text)
    if match:
        year, month = match.groups()
        return pd.Timestamp(year=int(year), month=int(month), day=1)

    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return ts
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _to_monthly_series(df: pd.DataFrame, date_col: str, value_col: str, name: str) -> pd.DataFrame:
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", name]
    out["date"] = out["date"].apply(parse_chinese_month)
    out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=["date", name])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out[["date", name]]


def fetch_pmi() -> pd.DataFrame:
    df = ak.macro_china_pmi()
    return _to_monthly_series(df, "月份", "制造业-指数", "PMI")


def fetch_cpi() -> pd.DataFrame:
    df = ak.macro_china_cpi()
    return _to_monthly_series(df, "月份", "全国-同比增长", "CPI")


def fetch_ppi() -> pd.DataFrame:
    df = ak.macro_china_ppi()
    return _to_monthly_series(df, "月份", "当月同比增长", "PPI")


def fetch_m2() -> pd.DataFrame:
    df = ak.macro_china_money_supply()
    return _to_monthly_series(df, "月份", "货币和准货币(M2)-同比增长", "M2")


# PBoC: end-2014 social financing stock = 122.86 trillion CNY (1228600 亿元)
TSF_STOCK_BASELINE_2014 = 1_228_600


def fetch_tsf_increments() -> pd.DataFrame:
    """Fetch monthly TSF increments (亿元) from AKShare."""
    df = ak.macro_china_shrzgm()
    tsf = df[["月份", "社会融资规模增量"]].copy()
    tsf["date"] = tsf["月份"].apply(parse_chinese_month)
    tsf["increment"] = pd.to_numeric(tsf["社会融资规模增量"], errors="coerce")
    tsf = tsf.dropna(subset=["date", "increment"]).sort_values("date")
    return tsf[["date", "increment"]]


def compute_tsf_stock_yoy(increments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TSF stock YoY from monthly increments.

    AKShare only provides increments from 2015-01; we anchor cumulative stock
    to the official end-2014 level before computing YoY growth.
    """
    tsf = increments.sort_values("date").copy()
    tsf["stock"] = TSF_STOCK_BASELINE_2014 + tsf["increment"].cumsum()
    tsf["TSF"] = tsf["stock"].pct_change(12) * 100
    return tsf[["date", "TSF"]].dropna(subset=["TSF"])


def fetch_tsf_stock_yoy() -> pd.DataFrame:
    increments = fetch_tsf_increments()
    return compute_tsf_stock_yoy(increments)


def _safe_fetch(name: str, fn, *args, **kwargs) -> pd.DataFrame | None:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"  [macro] skip {name}: {exc}")
        return None


def fetch_industrial_production_yoy() -> pd.DataFrame:
    df = ak.macro_china_gyzjz()
    col = "今值" if "今值" in df.columns else df.columns[-1]
    date_col = "日期" if "日期" in df.columns else "月份"
    return _to_monthly_series(df, date_col, col, "IND_PROD")


def fetch_fixed_investment_yoy() -> pd.DataFrame:
    df = ak.macro_china_gdzctz()
    col = "今值" if "今值" in df.columns else "当月"
    date_col = "日期" if "日期" in df.columns else "月份"
    return _to_monthly_series(df, date_col, col, "FIXED_INV")


def fetch_retail_sales_yoy() -> pd.DataFrame:
    df = ak.macro_china_consumer_goods_retail()
    col = "社会消费品零售总额-同比增长" if "社会消费品零售总额-同比增长" in df.columns else "今值"
    return _to_monthly_series(df, "月份", col, "RETAIL")


def fetch_export_yoy() -> pd.DataFrame:
    df = ak.macro_china_exports_yoy()
    col = "今值" if "今值" in df.columns else df.columns[-1]
    date_col = "日期" if "日期" in df.columns else "月份"
    return _to_monthly_series(df, date_col, col, "EXPORT_YOY")


def fetch_bond_10y_yield() -> pd.DataFrame:
    df = ak.bond_zh_us_rate()
    if "中国国债收益率10年" not in df.columns:
        return pd.DataFrame(columns=["date", "BOND_10Y"])
    out = df[["日期", "中国国债收益率10年"]].copy()
    out.columns = ["date", "BOND_10Y"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["BOND_10Y"] = pd.to_numeric(out["BOND_10Y"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out.groupby("date", as_index=False)["BOND_10Y"].last()


def fetch_fx_cny_monthly() -> pd.DataFrame:
    df = ak.currency_boc_sina(symbol="美元", start_date="20150101", end_date="20251231")
    out = df.rename(columns={"日期": "date", "中行汇买价": "FX_CNY"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["FX_CNY"] = pd.to_numeric(out["FX_CNY"], errors="coerce")
    out = out.dropna(subset=["date", "FX_CNY"]).sort_values("date")
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out.groupby("date", as_index=False)["FX_CNY"].last()


def fetch_unemployment_rate() -> pd.DataFrame:
    if not hasattr(ak, "macro_china_urban_unemployment"):
        return pd.DataFrame(columns=["date", "UNEMPLOY"])
    df = ak.macro_china_urban_unemployment()
    col = "失业率" if "失业率" in df.columns else df.columns[-1]
    date_col = "日期" if "日期" in df.columns else "月份"
    return _to_monthly_series(df, date_col, col, "UNEMPLOY")


def fetch_extended_macro() -> pd.DataFrame:
    """Optional macro series from public sources (NBS/PBoC/CFETS via AKShare)."""
    fetchers = [
        ("IND_PROD", fetch_industrial_production_yoy),
        ("FIXED_INV", fetch_fixed_investment_yoy),
        ("RETAIL", fetch_retail_sales_yoy),
        ("EXPORT_YOY", fetch_export_yoy),
        ("BOND_10Y", fetch_bond_10y_yield),
        ("FX_CNY", fetch_fx_cny_monthly),
        ("UNEMPLOY", fetch_unemployment_rate),
    ]
    extended = None
    for name, fn in fetchers:
        frame = _safe_fetch(name, fn)
        if frame is None or frame.empty:
            continue
        extended = frame if extended is None else extended.merge(frame, on="date", how="outer")
    return extended if extended is not None else pd.DataFrame(columns=["date"])


def fetch_all_macro(save: bool = True) -> pd.DataFrame:
    cfg = load_config()
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])

    frames = [fetch_pmi(), fetch_cpi(), fetch_ppi(), fetch_m2(), fetch_tsf_stock_yoy()]
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")

    extended = fetch_extended_macro()
    if not extended.empty and len(extended.columns) > 1:
        macro = macro.merge(extended, on="date", how="outer")

    macro = macro.sort_values("date")
    macro = macro[(macro["date"] >= start) & (macro["date"] <= end)]

    if save:
        raw_dir = get_path("raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        macro.to_csv(raw_dir / "macro_indicators.csv", index=False, encoding="utf-8-sig")
        try:
            increments = fetch_tsf_increments()
            increments.to_csv(raw_dir / "tsf_increments.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass

    return macro


if __name__ == "__main__":
    data = fetch_all_macro()
    print(data.head())
    print(data.tail())
    print(f"Shape: {data.shape}")
