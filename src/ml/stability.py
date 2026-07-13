"""Stability analysis: Top-K, forecast horizon, and training-window sensitivity."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.ml.metrics import compute_prediction_metrics
from src.ml.nested_walkforward import walk_forward_lambdarank_nested
from src.ml.strategy import evaluate_ml_strategy, run_ml_backtest
from src.ml.walkforward import walk_forward_lambdarank, walk_forward_panel_regression
from src.ml.metrics import compute_performance_metrics


def _default_weighting() -> str:
    return load_config().get("ml", {}).get("risk_control", {}).get("default_weighting", "softmax")


def _generate_predictions(
    df: pd.DataFrame,
    model_name: str,
    horizon: int = 1,
    train_mode: str = "expanding",
    rolling_window: int | None = None,
) -> pd.DataFrame:
    if model_name == "lambdarank":
        val_enabled = load_config().get("ml", {}).get("validation", {}).get("enabled", True)
        if val_enabled and horizon == 1 and train_mode == "expanding" and rolling_window is None:
            pred, _ = walk_forward_lambdarank_nested(df, horizon=horizon, train_mode=train_mode, rolling_window=rolling_window)
            return pred
        return walk_forward_lambdarank(
            df,
            horizon=horizon,
            train_mode=train_mode,
            rolling_window=rolling_window,
        )
    return walk_forward_panel_regression(
        df,
        model_name,
        horizon=horizon,
        train_mode=train_mode,
        rolling_window=rolling_window,
    )


def get_stability_config() -> dict:
    return load_config().get("ml", {}).get("stability", {})


def _summarize_run(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    top_k: int,
    analysis: str,
    variant: str,
    weighting: str | None = None,
) -> dict:
    weighting = weighting or _default_weighting()
    pred = predictions[predictions["model"] == model_name]
    if pred.empty:
        return {}

    metrics = compute_prediction_metrics(pred, top_k=top_k)
    bt, _ = run_ml_backtest(
        predictions,
        df,
        model_name,
        top_k=top_k,
        weighting=weighting,
        turnover_lambda=0.0,
    )
    perf = compute_performance_metrics(bt["portfolio_return"], model_name) if not bt.empty else {}

    ic_sum = metrics["ic_summary"]
    hit_sum = metrics["hit_rate_summary"]
    row = {
        "analysis": analysis,
        "variant": variant,
        "model": model_name,
        "top_k": top_k,
        "n_oos_months": len(bt),
        "annual_return": perf.get("annual_return"),
        "sharpe": perf.get("sharpe"),
        "max_drawdown": perf.get("max_drawdown"),
        "avg_turnover": float(bt["turnover"].mean()) if not bt.empty else None,
    }
    if not ic_sum.empty:
        ic_row = ic_sum[ic_sum["model"] == model_name]
        if not ic_row.empty:
            row["rank_ic_mean"] = float(ic_row["rank_ic_mean"].iloc[0])
            row["icir"] = float(ic_row["icir"].iloc[0])
    if not hit_sum.empty:
        hit_row = hit_sum[(hit_sum["model"] == model_name) & (hit_sum["top_k"] == top_k)]
        if not hit_row.empty:
            row["top_k_hit_rate"] = float(hit_row["hit_rate_mean"].iloc[0])
    if "horizon" in pred.columns:
        row["horizon"] = int(pred["horizon"].iloc[0])
    if "train_mode" in pred.columns:
        row["train_mode"] = pred["train_mode"].iloc[0]
        rw = pred["rolling_window"].iloc[0]
        if pd.notna(rw):
            row["rolling_window"] = int(rw)
    return row


def analyze_top_k_grid(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    top_k_list: list[int] | None = None,
) -> pd.DataFrame:
    cfg = get_stability_config()
    top_k_list = top_k_list or cfg.get("top_k_grid", [3, 5, 10])
    rows = [
        _summarize_run(predictions, df, model_name, k, "top_k", f"Top-{k}")
        for k in top_k_list
    ]
    return pd.DataFrame([r for r in rows if r])


def analyze_horizon_grid(
    df: pd.DataFrame,
    model_name: str,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    cfg = get_stability_config()
    horizons = horizons or cfg.get("horizon_grid", [1, 2, 3])
    rows: list[dict] = []
    for h in horizons:
        pred = _generate_predictions(df, model_name, horizon=h, train_mode="expanding")
        if pred.empty:
            continue
        top_k = load_config().get("ml", {}).get("top_k", 5)
        row = _summarize_run(pred, df, model_name, top_k, "horizon", f"{h}M forward")
        if row:
            row["horizon"] = h
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_train_window_grid(
    df: pd.DataFrame,
    model_name: str,
    windows: list[dict] | None = None,
) -> pd.DataFrame:
    cfg = get_stability_config()
    windows = windows or cfg.get(
        "train_windows",
        [
            {"mode": "expanding", "label": "Expanding"},
            {"mode": "rolling", "window": 36, "label": "Rolling 36M"},
            {"mode": "rolling", "window": 60, "label": "Rolling 60M"},
        ],
    )
    top_k = load_config().get("ml", {}).get("top_k", 5)
    rows: list[dict] = []
    for spec in windows:
        mode = spec.get("mode", "expanding")
        window = spec.get("window")
        label = spec.get("label", str(spec))
        pred = _generate_predictions(
            df,
            model_name,
            horizon=1,
            train_mode=mode,
            rolling_window=window,
        )
        if pred.empty:
            continue
        row = _summarize_run(pred, df, model_name, top_k, "train_window", label)
        if row:
            row["train_mode"] = mode
            if window:
                row["rolling_window"] = int(window)
            rows.append(row)
    return pd.DataFrame(rows)


def run_stability_analysis(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    model_name: str,
) -> dict[str, pd.DataFrame]:
    cfg = get_stability_config()
    if not cfg.get("enabled", True):
        return {}

    top_k = analyze_top_k_grid(predictions, df, model_name)
    horizon = analyze_horizon_grid(df, model_name)
    train_win = analyze_train_window_grid(df, model_name)

    out_dir = get_path("processed") / "ml" / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not top_k.empty:
        top_k.to_csv(out_dir / "stability_top_k.csv", index=False, encoding="utf-8-sig")
    if not horizon.empty:
        horizon.to_csv(out_dir / "stability_horizon.csv", index=False, encoding="utf-8-sig")
    if not train_win.empty:
        train_win.to_csv(out_dir / "stability_train_window.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat([top_k, horizon, train_win], ignore_index=True)
    if not combined.empty:
        combined.to_csv(out_dir / "stability_summary.csv", index=False, encoding="utf-8-sig")

    return {"top_k": top_k, "horizon": horizon, "train_window": train_win, "summary": combined}
