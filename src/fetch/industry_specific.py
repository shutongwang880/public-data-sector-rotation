"""Industry return-based public signals and price factors."""

from __future__ import annotations

import pandas as pd

from src.config import get_path
from src.data.align import filter_sample_range, to_month_start
from src.data.feature_catalog import PROXY_FEATURE_COLS, SIGNAL_FEATURE_COLS, normalize_proxy_columns
from src.data.industry_mask import get_comparable_industry_columns
from src.fetch.industry_price_signals import save_industry_price_signals

COMMODITY_SYMBOLS = {
    "钢铁": "RB0",
    "有色金属": "CU0",
    "煤炭": "JM0",
    "石油石化": "SC0",
    "基础化工": "MA0",
}


def _rolling_cum(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window, min_periods=max(2, window // 2)).apply(
        lambda x: x.prod() - 1, raw=True
    )


def _load_commodity_monthly_returns() -> pd.DataFrame:
    path = get_path("raw") / "commodity_monthly_returns.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


def fetch_commodity_monthly_returns(save: bool = True) -> pd.DataFrame:
    import akshare as ak

    path = get_path("raw") / "commodity_monthly_returns.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    symbols = sorted(set(COMMODITY_SYMBOLS.values()))
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            daily = ak.futures_main_sina(symbol=sym)
            daily["date"] = pd.to_datetime(daily["date"])
            daily = daily.sort_values("date")
            monthly = daily.set_index("date")["close"].resample("ME").last().pct_change()
            out = monthly.reset_index()
            out.columns = ["date", sym]
            out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
            frames.append(out)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    commodity = frames[0]
    for f in frames[1:]:
        commodity = commodity.merge(f, on="date", how="outer")
    commodity = filter_sample_range(commodity.sort_values("date"))

    if save:
        get_path("raw").mkdir(parents=True, exist_ok=True)
        commodity.to_csv(path, index=False, encoding="utf-8-sig")
    return commodity


def compute_proxy_features(
    industry_returns: pd.DataFrame,
    hs300_col: str = "HS300",
    commodity_returns: pd.DataFrame | None = None,
    ppi_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Return-based proxy variables for panel modeling.

    Paper terminology: 收益/价格/景气/库存 **代理变量** — not PE/PB or enterprise inventory.
    """
    df = to_month_start(industry_returns)
    industries = get_comparable_industry_columns(df)
    hs300 = df[hs300_col] if hs300_col in df.columns else pd.Series(0.0, index=df.index)
    hs300_12 = _rolling_cum(hs300, 12)

    mom_3 = pd.DataFrame({ind: _rolling_cum(df[ind], 3) for ind in industries})
    mom_12 = pd.DataFrame({ind: _rolling_cum(df[ind], 12) for ind in industries})
    mom_cs = mom_3.rank(axis=1, pct=True)

    comm_idx = (
        commodity_returns.set_index("date")
        if commodity_returns is not None and not commodity_returns.empty
        else None
    )
    ppi_idx = None
    if ppi_series is not None and not ppi_series.empty:
        ppi_idx = ppi_series.copy()
        if not isinstance(ppi_idx.index, pd.DatetimeIndex):
            ppi_idx.index = pd.to_datetime(ppi_idx.index)

    rows: list[dict] = []
    for industry in industries:
        rel_mom = mom_12[industry] - hs300_12
        ret_accel = mom_3[industry] - mom_12[industry] / 4.0

        for i, row in df.iterrows():
            if pd.isna(df.loc[i, industry]):
                continue

            commodity_ret = float("nan")
            if industry in COMMODITY_SYMBOLS and comm_idx is not None:
                sym = COMMODITY_SYMBOLS[industry]
                if sym in comm_idx.columns and row["date"] in comm_idx.index:
                    commodity_ret = comm_idx.loc[row["date"], sym]
            if pd.isna(commodity_ret) and industry in COMMODITY_SYMBOLS and ppi_idx is not None:
                dt = row["date"]
                if dt in ppi_idx.index:
                    commodity_ret = float(ppi_idx.loc[dt]) / 100.0

            rows.append(
                {
                    "date": row["date"],
                    "industry": industry,
                    "IND_REL_MOM_12M": rel_mom.loc[i],
                    "IND_RET_ACCEL": ret_accel.loc[i],
                    "IND_MOM_CS_PCT": mom_cs.loc[i, industry],
                    "IND_COMMODITY": commodity_ret,
                }
            )

    out = pd.DataFrame(rows)
    return out.dropna(subset=["IND_REL_MOM_12M", "IND_RET_ACCEL", "IND_MOM_CS_PCT"]).reset_index(
        drop=True
    )


def save_proxy_features(
    industry_returns: pd.DataFrame,
    hs300_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = to_month_start(industry_returns)
    if hs300_returns is not None:
        hs300_returns = to_month_start(hs300_returns)
        merged = merged.merge(hs300_returns, on="date", how="left")

    try:
        commodity = fetch_commodity_monthly_returns(save=True)
    except Exception:
        commodity = _load_commodity_monthly_returns()
    if commodity is None or commodity.empty:
        commodity = None

    ppi_series = None
    universe_path = get_path("raw") / "feature_universe.csv"
    if universe_path.exists():
        universe = pd.read_csv(universe_path, parse_dates=["date"])
        universe = to_month_start(universe)
        if "PPI" in universe.columns:
            ppi_series = universe.set_index("date")["PPI"]

    features = compute_proxy_features(merged, commodity_returns=commodity, ppi_series=ppi_series)
    path = get_path("raw") / "feature_industry_proxies.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False, encoding="utf-8-sig")

    # Legacy filename for backward compatibility
    legacy = get_path("raw") / "feature_industry_specific.csv"
    features.to_csv(legacy, index=False, encoding="utf-8-sig")
    return features


def save_all_industry_features(
    industry_returns: pd.DataFrame,
    hs300_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Phase 1: price signals + public signals + fundamentals + industry drivers."""
    from src.fetch.industry_drivers import save_industry_drivers
    from src.fetch.industry_fundamentals import save_industry_fundamentals

    save_industry_price_signals(industry_returns, hs300_returns, save=True)
    save_proxy_features(industry_returns, hs300_returns)
    save_industry_fundamentals(industry_returns)
    save_industry_drivers(industry_returns)
    return load_merged_industry_features()


def save_industry_specific_features(
    industry_returns: pd.DataFrame,
    hs300_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Backward-compatible entry point."""
    return save_all_industry_features(industry_returns, hs300_returns)


# Legacy export
INDUSTRY_SPECIFIC_COLS = SIGNAL_FEATURE_COLS
compute_industry_specific_features = compute_proxy_features


def load_merged_industry_features() -> pd.DataFrame:
    """Load proxies + fundamentals + drivers; normalize legacy column names."""
    raw = get_path("raw")
    parts: list[pd.DataFrame] = []

    for name in ["feature_industry_proxies.csv", "feature_industry_specific.csv"]:
        path = raw / name
        if path.exists():
            df = normalize_proxy_columns(pd.read_csv(path, parse_dates=["date"]))
            parts.append(df)
            break

    price_path = raw / "feature_industry_price.csv"
    if price_path.exists():
        parts.append(pd.read_csv(price_path, parse_dates=["date"]))

    for name in ["feature_industry_fundamentals.csv", "feature_industry_drivers.csv"]:
        path = raw / name
        if path.exists():
            parts.append(pd.read_csv(path, parse_dates=["date"]))

    if not parts:
        return pd.DataFrame()

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["date", "industry"], how="left")
    return merged
