"""Factor attribution: momentum-only vs macro vs liquidity vs drivers."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.data.catalog import zscore_col
from src.data.feature_selection import (
    DRIVER_BLOCK,
    LIQUIDITY_MARKET_BLOCK,
    MACRO_BLOCK,
    MOMENTUM_GLOBAL,
    MOMENTUM_PANEL,
    get_selected_global_raw,
    get_selected_industry_panel,
)
from src.ml.data import load_ml_dataset
from src.ml.panel import build_long_panel
from src.ml.strategy import evaluate_ml_strategy
from src.ml.walkforward import walk_forward_panel_regression

FACTOR_ATTRIBUTION_STAGES: list[tuple[str, str, list[str], list[str]]] = [
    (
        "momentum_only",
        "Momentum Only",
        MOMENTUM_GLOBAL,
        MOMENTUM_PANEL,
    ),
    (
        "macro_only",
        "Macro Only",
        MACRO_BLOCK,
        [],
    ),
    (
        "macro_liquidity",
        "Macro + Liquidity/Market",
        MACRO_BLOCK + LIQUIDITY_MARKET_BLOCK,
        [],
    ),
    (
        "full_no_momentum_panel",
        "Macro/Liq + Drivers (no extra price panel)",
        get_selected_global_raw(),
        DRIVER_BLOCK,
    ),
    (
        "full_curated",
        "Full Curated Model",
        get_selected_global_raw(),
        get_selected_industry_panel(),
    ),
]


def _resolve_panel_columns(
    panel: pd.DataFrame,
    global_names: list[str],
    industry_names: list[str],
) -> list[str]:
    cols: list[str] = []
    for name in global_names:
        zname = zscore_col(name)
        if zname in panel.columns and panel[zname].notna().any():
            cols.append(zname)
        elif name in panel.columns and panel[name].notna().any():
            cols.append(name)
    for name in industry_names:
        if name in panel.columns and panel[name].notna().any():
            cols.append(name)
    if "industry_id" in panel.columns:
        cols.append("industry_id")
    return cols


def run_factor_attribution(
    df: pd.DataFrame | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    cfg = load_config().get("ml", {})
    model_name = model_name or cfg.get("ablation_model", "lasso")
    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    panel = build_long_panel(df)
    if panel.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for stage_id, label, global_feats, industry_feats in FACTOR_ATTRIBUTION_STAGES:
        feat_cols = _resolve_panel_columns(panel, global_feats, industry_feats)
        if len(feat_cols) <= 1:
            continue

        preds = walk_forward_panel_regression(df, model_name, feature_cols=feat_cols)
        if preds.empty:
            continue

        perf = evaluate_ml_strategy(
            preds,
            df,
            model_name,
            weighting="equal",
            turnover_lambda=0.0,
        )["performance"]
        rows.append(
            {
                "stage_id": stage_id,
                "stage_label": label,
                "n_features": len(feat_cols),
                "annual_return": perf.get("annual_return"),
                "sharpe": perf.get("sharpe"),
                "rank_ic_mean": perf.get("rank_ic_mean"),
                "top_k_hit_rate": perf.get("top_k_hit_rate"),
            }
        )

    out = pd.DataFrame(rows)
    out_dir = get_path("processed") / "ml" / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "factor_attribution.csv", index=False, encoding="utf-8-sig")
    return out
