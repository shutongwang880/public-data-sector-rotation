"""Ablation study: cumulative feature blocks vs Sharpe."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.data.catalog import zscore_col
from src.data.feature_catalog import ABLATION_STAGES
from src.ml.data import load_ml_dataset
from src.ml.panel import build_long_panel
from src.ml.strategy import evaluate_ml_strategy
from src.ml.walkforward import walk_forward_panel_regression


def _resolve_panel_columns(panel: pd.DataFrame, base_names: list[str]) -> list[str]:
    """Map raw feature names to panel columns (zscore + industry cols with data)."""
    cols: list[str] = []
    for name in base_names:
        zname = zscore_col(name)
        if zname in panel.columns and panel[zname].notna().any():
            cols.append(zname)
        elif name in panel.columns and panel[name].notna().any():
            cols.append(name)
    if "industry_id" in panel.columns:
        cols.append("industry_id")
    return cols


def run_ablation_study(
    df: pd.DataFrame | None = None,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Cumulative ablation: macro → market → industry global → proxy → driver.

    Uses the same walk-forward + Top-K backtest as the main pipeline.
    """
    cfg = load_config().get("ml", {})
    model_name = cfg.get("ablation_model", model_name)
    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    panel = build_long_panel(df)
    if panel.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for stage_id, label, base_feats in ABLATION_STAGES:
        feat_cols = _resolve_panel_columns(panel, base_feats)
        if len(feat_cols) <= 1:
            continue

        preds = walk_forward_panel_regression(df, model_name, feature_cols=feat_cols)
        if preds.empty:
            continue

        perf = evaluate_ml_strategy(preds, df, model_name)["performance"]
        rows.append(
            {
                "stage_id": stage_id,
                "stage_label": label,
                "n_features": len(feat_cols),
                "annual_return": perf.get("annual_return"),
                "sharpe": perf.get("sharpe"),
                "max_drawdown": perf.get("max_drawdown"),
                "rank_ic_mean": perf.get("rank_ic_mean"),
                "top_k_hit_rate": perf.get("top_k_hit_rate"),
            }
        )

    out = pd.DataFrame(rows)
    out_dir = get_path("processed") / "ml" / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "ablation_study.csv", index=False, encoding="utf-8-sig")
    return out
