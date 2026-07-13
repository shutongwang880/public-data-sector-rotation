"""Industry fundamentals: PE, PB, ROE, consensus EPS (Step 1)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import get_path, load_config
from src.data.align import filter_sample_range, to_month_start
from src.data.feature_catalog import FUNDAMENTAL_COLS
from src.data.industry_mask import get_comparable_industry_columns

# SW L1 code -> industry name (subset used by Tushare index_code mapping)
SW_CODE_BY_INDUSTRY: dict[str, str] = {
    "农林牧渔": "801010",
    "基础化工": "801030",
    "钢铁": "801040",
    "有色金属": "801050",
    "电子": "801080",
    "汽车": "801880",
    "家用电器": "801110",
    "食品饮料": "801120",
    "纺织服饰": "801130",
    "轻工制造": "801140",
    "医药生物": "801150",
    "公用事业": "801160",
    "交通运输": "801170",
    "房地产": "801180",
    "商贸零售": "801200",
    "社会服务": "801210",
    "综合": "801230",
    "建筑材料": "801710",
    "建筑装饰": "801720",
    "电力设备": "801730",
    "国防军工": "801740",
    "计算机": "801750",
    "传媒": "801760",
    "通信": "801770",
    "银行": "801780",
    "非银金融": "801790",
    "煤炭": "801950",
    "石油石化": "801960",
    "环保": "801970",
    "美容护理": "801980",
}


def _manual_fundamentals_path() -> Path:
    return get_path("raw") / "manual" / "industry_fundamentals.csv"


def load_manual_fundamentals() -> pd.DataFrame:
    """Optional Wind/Choice export: date, industry, IND_PE, IND_PB, IND_ROE, IND_EPS_FY1."""
    path = _manual_fundamentals_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return to_month_start(df)


def _get_tushare_token() -> str | None:
    cfg = load_config()
    token = cfg.get("data", {}).get("tushare_token") or os.environ.get("TUSHARE_TOKEN")
    return token.strip() if token else None


def fetch_fundamentals_tushare(dates: pd.Series | None = None) -> pd.DataFrame:
    """
    Fetch SW industry consensus fundamentals via Tushare Pro `con_forecast_sw`.

    Requires TUSHARE_TOKEN env or config `data.tushare_token`.
    """
    token = _get_tushare_token()
    if not token:
        return pd.DataFrame()

    try:
        import tushare as ts
    except ImportError:
        print("  [fundamentals] tushare not installed; pip install tushare")
        return pd.DataFrame()

    pro = ts.pro_api(token)
    rows: list[dict] = []

    if dates is None:
        cfg = load_config()
        start = pd.Timestamp(cfg["data"]["start_date"])
        end = pd.Timestamp(cfg["data"]["end_date"])
        dates = pd.date_range(start, end, freq="MS")

    for dt in dates:
        trade_date = dt.strftime("%Y%m%d")
        for industry, code in SW_CODE_BY_INDUSTRY.items():
            try:
                chunk = pro.query(
                    "con_forecast_sw",
                    code_idx=code,
                    trade_date=trade_date,
                )
            except Exception:
                continue
            if chunk is None or chunk.empty:
                continue
            row = chunk.sort_values("con_date").iloc[-1]
            rows.append(
                {
                    "date": pd.Timestamp(year=dt.year, month=dt.month, day=1),
                    "industry": industry,
                    "IND_PE": row.get("con_pe"),
                    "IND_PB": row.get("con_pb"),
                    "IND_ROE": row.get("con_roe"),
                    "IND_EPS_FY1": row.get("con_eps"),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    for col in FUNDAMENTAL_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return filter_sample_range(to_month_start(out))


def save_industry_fundamentals(
    industry_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and cache industry fundamentals long table."""
    path = get_path("raw") / "feature_industry_fundamentals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    manual = load_manual_fundamentals()
    tushare_df = fetch_fundamentals_tushare()

    frames = [f for f in [manual, tushare_df] if f is not None and not f.empty]
    if not frames:
        print("  [fundamentals] no PE/PB/ROE/EPS source; set TUSHARE_TOKEN or add manual CSV")
        if industry_returns is not None:
            industries = get_comparable_industry_columns(to_month_start(industry_returns))
            dates = to_month_start(industry_returns)["date"].unique()
            empty = pd.MultiIndex.from_product([dates, industries], names=["date", "industry"])
            out = empty.to_frame(index=False)
            for col in FUNDAMENTAL_COLS:
                out[col] = float("nan")
            out.to_csv(path, index=False, encoding="utf-8-sig")
            return out
        pd.DataFrame(columns=["date", "industry", *FUNDAMENTAL_COLS]).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = to_month_start(combined)
    combined = combined.sort_values(["date", "industry"]).drop_duplicates(
        ["date", "industry"], keep="last"
    )
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined
