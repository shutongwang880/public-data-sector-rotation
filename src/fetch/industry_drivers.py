"""Industry-specific driver variables (Step 2): sparse by industry."""

from __future__ import annotations

import pandas as pd

from src.config import get_path
from src.data.align import filter_sample_range, to_month_start
from src.data.feature_catalog import DRIVER_COLS, INDUSTRY_DRIVER_MAP
from src.data.industry_mask import get_comparable_industry_columns


def _monthly_from_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.sort_values(date_col).set_index(date_col)[value_col]
    monthly = s.resample("MS").last()
    return monthly.pct_change()


def fetch_macro_driver_panel() -> pd.DataFrame:
    """
    Monthly macro series mapped to driver columns.

    All are proxies — see feature_catalog.DRIVER_DESCRIPTIONS.
    """
    import akshare as ak

    frames: list[pd.DataFrame] = []

    # Base monthly grid from PPI (coal price + inventory cycle PROXY)
    universe_path = get_path("raw") / "feature_universe.csv"
    if universe_path.exists():
        uni = pd.read_csv(universe_path, parse_dates=["date"])
        uni = to_month_start(uni)
        if "PPI" in uni.columns:
            base = uni[["date", "PPI"]].copy()
            base["IND_DRV_COAL_PRICE"] = base["PPI"] / 100.0
            base["IND_DRV_INV_PROXY"] = base["PPI"].diff()
            frames.append(base[["date", "IND_DRV_COAL_PRICE", "IND_DRV_INV_PROXY"]])

    # Semiconductor: SOX index monthly change
    try:
        sox = ak.macro_global_sox_index()
        sox["date"] = pd.to_datetime(sox["日期"])
        sox = sox.sort_values("date").drop_duplicates("date", keep="last")
        sox["date"] = sox["date"].dt.to_period("M").dt.to_timestamp()
        sox["IND_DRV_SEMI"] = pd.to_numeric(sox["最新值"], errors="coerce").pct_change()
        frames.append(sox[["date", "IND_DRV_SEMI"]])
    except Exception as exc:
        print(f"  [drivers] SOX fetch failed: {exc}")

    # Bank spread proxy: LPR1Y - deposit benchmark (RATE_1)
    try:
        lpr = ak.macro_china_lpr()
        lpr["date"] = pd.to_datetime(lpr["TRADE_DATE"])
        lpr = lpr.sort_values("date").drop_duplicates("date", keep="last")
        lpr["date"] = lpr["date"].dt.to_period("M").dt.to_timestamp()
        lpr["IND_DRV_BANK_SPREAD"] = pd.to_numeric(lpr["LPR1Y"], errors="coerce") - pd.to_numeric(
            lpr["RATE_1"], errors="coerce"
        )
        frames.append(lpr[["date", "IND_DRV_BANK_SPREAD"]])
    except Exception as exc:
        print(f"  [drivers] LPR spread fetch failed: {exc}")

    # Housing: national real-estate climate index level
    try:
        re_idx = ak.macro_china_real_estate()
        re_idx["date"] = pd.to_datetime(re_idx["日期"])
        re_idx = re_idx.sort_values("date").drop_duplicates("date", keep="last")
        re_idx["date"] = re_idx["date"].dt.to_period("M").dt.to_timestamp()
        re_idx["IND_DRV_HOUSING"] = pd.to_numeric(re_idx["最新值"], errors="coerce")
        frames.append(re_idx[["date", "IND_DRV_HOUSING"]])
    except Exception as exc:
        print(f"  [drivers] real estate index fetch failed: {exc}")

    if not frames:
        return pd.DataFrame(columns=["date", *DRIVER_COLS])

    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on="date", how="outer")
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp()
    panel = panel.drop_duplicates("date", keep="last")
    panel = filter_sample_range(panel.sort_values("date"))
    for col in DRIVER_COLS:
        if col not in panel.columns:
            panel[col] = float("nan")
    return panel


def expand_drivers_to_industries(
    driver_panel: pd.DataFrame,
    industries: list[str],
    dates: pd.Series,
) -> pd.DataFrame:
    """Expand macro drivers to long (date, industry); non-applicable industries get NaN."""
    rows: list[dict] = []
    driver_idx = driver_panel.set_index("date") if not driver_panel.empty else None

    for dt in dates:
        dt = pd.Timestamp(dt)
        for industry in industries:
            row: dict = {"date": dt, "industry": industry}
            for drv in DRIVER_COLS:
                val = float("nan")
                if (
                    driver_idx is not None
                    and drv in driver_idx.columns
                    and dt in driver_idx.index
                    and industry in INDUSTRY_DRIVER_MAP.get(drv, [])
                ):
                    raw = driver_idx.loc[dt, drv]
                    val = float(raw.iloc[0]) if isinstance(raw, pd.Series) else float(raw)
                row[drv] = val
            rows.append(row)

    return pd.DataFrame(rows)


def save_industry_drivers(industry_returns: pd.DataFrame) -> pd.DataFrame:
    """Fetch macro drivers and expand to industry long table."""
    path = get_path("raw") / "feature_industry_drivers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df = to_month_start(industry_returns)
    industries = get_comparable_industry_columns(df)
    driver_panel = fetch_macro_driver_panel()
    if driver_panel.empty and path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        if not cached.empty:
            return cached
    long = expand_drivers_to_industries(driver_panel, industries, df["date"])
    long.to_csv(path, index=False, encoding="utf-8-sig")
    return long
