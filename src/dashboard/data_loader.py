"""Load processed outputs for dashboard and business analytics views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_path, load_config


def _read_csv(rel: str) -> pd.DataFrame | None:
    path = get_path("processed") / rel
    if not path.exists():
        return None
    cols = pd.read_csv(path, nrows=0).columns
    parse = [c for c in cols if c.lower() == "date" or c in ("feature_date", "target_date")]
    return pd.read_csv(path, parse_dates=parse if parse else None)


def _month_label(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "—"
    t = pd.Timestamp(ts)
    return t.strftime("%b %Y")


def _softmax_weights(scores: np.ndarray, temperature: float = 1.0, cap: float = 0.20) -> np.ndarray:
    if len(scores) == 0:
        return np.array([])
    s = scores / max(temperature, 1e-6)
    exp_s = np.exp(s - s.max())
    w = exp_s / exp_s.sum() if exp_s.sum() > 0 else np.ones(len(scores)) / len(scores)
    w = np.minimum(w, cap)
    return w / w.sum() if w.sum() > 0 else w


def load_primary_metrics() -> dict:
    path = get_path("processed") / "ml" / "model_comparison_primary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        k: (float(row[k]) if pd.notna(row.get(k)) and isinstance(row.get(k), (int, float)) else row.get(k))
        for k in row.index
    }


def load_benchmarks() -> pd.DataFrame:
    df = _read_csv("ml/benchmark_comparison.csv")
    return df if df is not None else pd.DataFrame()


def load_backtest() -> pd.DataFrame:
    df = _read_csv("ml/backtest_lambdarank.csv")
    return df if df is not None else pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    for name in ["ml/predictions_lambdarank_nested.csv", "ml/predictions.csv"]:
        df = _read_csv(name)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()


def load_ic_decay() -> pd.DataFrame:
    df = _read_csv("ml/analysis/ic_decay_summary.csv")
    return df if df is not None else pd.DataFrame()


def load_error_summary() -> pd.DataFrame:
    df = _read_csv("ml/analysis/error_analysis_summary.csv")
    return df if df is not None else pd.DataFrame()


def load_case_studies() -> pd.DataFrame:
    path = get_path("processed") / "ml" / "case_studies" / "case_study_shap.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def _pipeline_data_cap() -> dict:
    """How far the training panel can go (master_dataset_train is the bottleneck)."""
    master_path = get_path("processed") / "master_dataset_train.csv"
    feat_path = get_path("processed") / "features_standardized.csv"
    out = {"master_max": None, "features_max": None, "bottleneck": None}
    if master_path.exists():
        m = pd.read_csv(master_path, parse_dates=["date"])
        out["master_max"] = m["date"].max()
    if feat_path.exists():
        f = pd.read_csv(feat_path, parse_dates=["date"])
        out["features_max"] = f["date"].max()
    if out["master_max"] is not None and out["features_max"] is not None:
        if out["features_max"] > out["master_max"]:
            out["bottleneck"] = (
                f"特征数据已到 {_month_label(out['features_max'])}，"
                f"但训练主表仅到 {_month_label(out['master_max'])}"
                f"（通常因最新月宏观 M2/社融 尚未发布完整，dropna 后无法入库）"
            )
    return out


def load_live_forecast(predictions: pd.DataFrame | None = None) -> dict:
    """
    Latest-month industry forecast for dashboard (product view).

    Uses the most recent row in predictions CSV — i.e. model output after run_ml.py.
    """
    pred = predictions if predictions is not None else load_predictions()
    if pred.empty:
        return {}

    cfg = load_config().get("ml", {})
    top_k = int(cfg.get("top_k", 5))
    temperature = float(cfg.get("risk_control", {}).get("softmax_temperature", 1.0))
    max_w = float(cfg.get("risk_control", {}).get("max_industry_weight", 0.20))

    if "model" in pred.columns:
        pred = pred[pred["model"] == cfg.get("primary_model", "lambdarank")]
    if pred.empty:
        return {}

    latest_date = pred["date"].max()
    month = pred[pred["date"] == latest_date].sort_values("y_pred", ascending=False).reset_index(drop=True)
    month["rank"] = month.index + 1
    n = len(month)

    month["signal"] = "Neutral"
    month.loc[month["rank"] <= top_k, "signal"] = "Overweight"
    month.loc[month["rank"] > n - 5, "signal"] = "Underweight"

    top = month.head(top_k)
    weights = _softmax_weights(top["y_pred"].to_numpy(dtype=float), temperature, max_w)

    rankings = []
    for i, row in month.iterrows():
        weight = None
        if row["rank"] <= top_k:
            weight = float(weights[int(row["rank"]) - 1])
        rankings.append(
            {
                "rank": int(row["rank"]),
                "industry": row["industry"],
                "score": float(row["y_pred"]),
                "signal": row["signal"],
                "weight": weight,
                "actual_1m": float(row["y_actual"]) if pd.notna(row.get("y_actual")) else None,
            }
        )

    meta = month.iloc[0]
    feature_date = meta.get("feature_date", latest_date)
    target_date = meta.get("target_date", latest_date)

    pred_path = get_path("processed") / "ml" / "predictions_lambdarank_nested.csv"
    updated_at = pd.Timestamp(pred_path.stat().st_mtime, unit="s") if pred_path.exists() else pd.Timestamp(latest_date)

    from datetime import datetime

    today = datetime.now()
    feat_ts = pd.Timestamp(feature_date)
    target_ts = pd.Timestamp(target_date)
    # 月频策略：预测目标月落后当前 1–2 个月属正常（宏观发布滞后 + 月末决策）
    forecast_months_behind = (today.year - target_ts.year) * 12 + (today.month - target_ts.month)
    data_months_behind = (today.year - feat_ts.year) * 12 + (today.month - feat_ts.month)

    cap = _pipeline_data_cap()
    is_stale = forecast_months_behind > 2

    staleness_note = None
    if is_stale:
        parts = [
            f"最新预测目标月为 {_month_label(target_date)}，距当前约 {max(0, forecast_months_behind)} 个月。"
        ]
        if cap.get("bottleneck"):
            parts.append(cap["bottleneck"])
        else:
            parts.append("可尝试重新运行 python run_phase1.py && python run_ml.py。")
        staleness_note = "".join(parts)
    elif cap.get("bottleneck"):
        staleness_note = None  # 不在 Dashboard 展示技术瓶颈说明

    return {
        "forecast_month": _month_label(target_date),
        "forecast_month_raw": str(target_ts.date()) if pd.notna(target_date) else None,
        "data_as_of": _month_label(feature_date),
        "data_as_of_raw": str(feat_ts.date()) if pd.notna(feature_date) else None,
        "prediction_date": str(pd.Timestamp(latest_date).date()),
        "updated_at": updated_at.strftime("%Y-%m-%d %H:%M"),
        "today": today.strftime("%Y-%m-%d"),
        "months_stale": max(0, forecast_months_behind),
        "feature_months_behind": max(0, data_months_behind),
        "pipeline_cap": cap,
        "is_stale": is_stale,
        "is_pipeline_latest": not is_stale,
        "staleness_note": staleness_note,
        "model": cfg.get("primary_model", "lambdarank"),
        "top_k": top_k,
        "n_industries": n,
        "rankings": rankings,
        "top_holdings": [r for r in rankings if r["signal"] == "Overweight"],
        "avoid": [r for r in rankings if r["signal"] == "Underweight"],
        "headline": (
            f"Using data through {_month_label(feature_date)}, "
            f"the model ranks sector relative performance for {_month_label(target_date)}."
        ),
    }


def load_latest_recommendation(predictions: pd.DataFrame | None = None) -> dict:
    """Backward-compatible wrapper around live forecast."""
    fc = load_live_forecast(predictions)
    if not fc:
        return {}
    return {
        "date": fc.get("prediction_date"),
        "month_label": fc.get("forecast_month"),
        "headline": fc.get("headline"),
        "data_as_of": fc.get("data_as_of"),
        "holdings": fc.get("top_holdings", []),
        "avoid": fc.get("avoid", []),
        "n_industries_scored": fc.get("n_industries", 0),
        "updated_at": fc.get("updated_at"),
    }


def parse_weights_string(weights_str: str) -> list[dict]:
    if not isinstance(weights_str, str) or not weights_str:
        return []
    items = []
    for part in weights_str.split(","):
        if ":" not in part:
            continue
        name, w = part.split(":", 1)
        items.append({"industry": name.strip(), "weight": float(w)})
    return items

