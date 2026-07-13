"""Build long-format panel data for pooled supervised learning."""

from __future__ import annotations

import pandas as pd

from src.config import get_path
from src.data.align import to_month_start
from src.data.catalog import ALL_FEATURE_COLS
from src.data.feature_catalog import (
    ALL_INDUSTRY_FEATURE_COLS,
    normalize_proxy_columns,
)
from src.data.feature_selection import get_selected_industry_panel
from src.data.industry_mask import get_comparable_industry_columns
from src.fetch.industry_specific import compute_proxy_features, load_merged_industry_features
from src.ml.data import get_feature_columns, get_ml_config, get_ml_industries, load_ml_dataset


def ensure_industry_features(df: pd.DataFrame) -> pd.DataFrame:
    cached = load_merged_industry_features()
    if not cached.empty:
        return cached

    ind_cols = get_comparable_industry_columns(df)
    if not ind_cols:
        ind_cols = [
            c
            for c in df.columns
            if c not in {"Date", "date", "HS300", *ALL_FEATURE_COLS}
            and not c.endswith("_zscore")
        ]
    industry_df = df[["date", *ind_cols]].copy()
    if "HS300" in df.columns:
        industry_df["HS300"] = df["HS300"]
    else:
        hs = pd.read_csv(get_path("raw") / "hs300_monthly_returns.csv", parse_dates=["date"])
        industry_df = industry_df.merge(to_month_start(hs), on="date", how="left")
    return normalize_proxy_columns(compute_proxy_features(industry_df))


def get_panel_feature_columns(
    panel: pd.DataFrame,
    include_industry_id: bool = True,
) -> list[str]:
    cfg = get_ml_config()
    global_cols = get_feature_columns(panel)
    industry_cols = [
        c
        for c in get_selected_industry_panel()
        if c in panel.columns and panel[c].notna().any()
    ]
    cols = global_cols + industry_cols
    if include_industry_id and "industry_id" in panel.columns and cfg.get("use_industry_id", True):
        cols = cols + ["industry_id"]
    return cols


def _compound_industry_return(df: pd.DataFrame, industry: str, start_idx: int, horizon: int) -> float:
    """Compound monthly industry returns from start_idx over `horizon` months."""
    prod = 1.0
    valid = 0
    for j in range(start_idx, min(start_idx + horizon, len(df))):
        r = df.iloc[j][industry]
        if pd.isna(r):
            return float("nan")
        prod *= 1.0 + float(r)
        valid += 1
    if valid < horizon:
        return float("nan")
    return prod - 1.0


def build_long_panel(
    df: pd.DataFrame | None = None,
    industries: list[str] | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Stack (date, industry) long table: X_t + industry features -> forward return over `horizon` months."""
    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    industries = industries or get_ml_industries(df)
    global_cols = get_feature_columns(df)
    ind_feat = ensure_industry_features(df)
    horizon = max(1, int(horizon))

    ppi_fallback = df.set_index("date")["PPI"] if "PPI" in df.columns else None
    industry_cols = [c for c in get_selected_industry_panel() if c in ind_feat.columns]

    rows: list[dict] = []
    industry_to_id = {name: i for i, name in enumerate(industries)}

    for i in range(len(df) - horizon):
        x_row = df.iloc[i]
        decision_date = df.iloc[i + 1]["date"]
        target_date = df.iloc[i + horizon]["date"]
        feats = x_row[global_cols]
        if feats.isna().any():
            continue

        for industry in industries:
            y_fwd = _compound_industry_return(df, industry, i + 1, horizon)
            if pd.isna(y_fwd):
                continue

            spec = ind_feat[(ind_feat["date"] == x_row["date"]) & (ind_feat["industry"] == industry)]
            if spec.empty:
                continue
            spec_row = spec.iloc[0]

            row = {
                "Date": x_row.get("Date", x_row["date"].strftime("%Y-%m")),
                "date": decision_date,
                "feature_date": x_row["date"],
                "target_date": target_date,
                "horizon": horizon,
                "industry": industry,
                "industry_id": industry_to_id[industry],
                "y": float(y_fwd),
                **{c: float(feats[c]) for c in global_cols},
            }

            for col in industry_cols:
                val = spec_row.get(col)
                if col == "IND_COMMODITY" and pd.isna(val) and industry in {
                    "钢铁",
                    "有色金属",
                    "煤炭",
                    "石油石化",
                    "基础化工",
                }:
                    if ppi_fallback is not None and x_row["date"] in ppi_fallback.index:
                        val = float(ppi_fallback.loc[x_row["date"]]) / 100.0
                row[col] = float(val) if pd.notna(val) else float("nan")

            rows.append(row)

    panel = pd.DataFrame(rows)
    return panel.sort_values(["target_date", "industry"]).reset_index(drop=True)
