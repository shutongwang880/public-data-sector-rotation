"""Backtest and prediction evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import load_config

TRADING_MONTHS = 12


def compute_performance_metrics(returns: pd.Series, name: str) -> dict[str, float | str]:
    """Compute standard backtest metrics from a monthly return series."""
    cfg = load_config().get("strategy", {})
    rf_annual = cfg.get("risk_free_rate", 0.02)
    rf_monthly = (1 + rf_annual) ** (1 / TRADING_MONTHS) - 1

    rets = returns.dropna()
    if rets.empty:
        return {"name": name}

    nav = (1 + rets).cumprod()
    total_return = nav.iloc[-1] - 1
    n_months = len(rets)
    annual_return = (1 + total_return) ** (TRADING_MONTHS / n_months) - 1
    volatility = float(rets.std() * np.sqrt(TRADING_MONTHS))
    sharpe = (
        float((rets.mean() - rf_monthly) / rets.std() * np.sqrt(TRADING_MONTHS))
        if rets.std() > 0
        else np.nan
    )
    max_dd = _max_drawdown(nav)
    win_rate = float((rets > 0).mean())
    calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else np.nan

    return {
        "name": name,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "months": int(n_months),
    }


def monthly_rank_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank IC per forecast month."""
    rows: list[dict] = []
    for date, group in predictions.groupby("date"):
        sub = group.dropna(subset=["y_actual", "y_pred"])
        if len(sub) < 3:
            continue
        ic, _ = spearmanr(sub["y_pred"], sub["y_actual"])
        rows.append({"date": date, "rank_ic": float(ic) if pd.notna(ic) else np.nan})
    return pd.DataFrame(rows)


def top_k_hit_rate(predictions: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Fraction of predicted Top-K industries that appear in realised Top-K."""
    rows: list[dict] = []
    for date, group in predictions.groupby("date"):
        sub = group.dropna(subset=["y_actual", "y_pred"])
        if len(sub) < top_k:
            continue
        pred_top = set(sub.nlargest(top_k, "y_pred")["industry"])
        actual_top = set(sub.nlargest(top_k, "y_actual")["industry"])
        rows.append({"date": date, "hit_rate": len(pred_top & actual_top) / top_k})
    return pd.DataFrame(rows)


def compute_prediction_metrics(
    predictions: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, pd.DataFrame]:
    """Aggregate IC and Top-K hit summaries by model."""
    if predictions.empty:
        return {"ic_summary": pd.DataFrame(), "hit_rate_summary": pd.DataFrame()}

    ic_rows: list[dict] = []
    hit_rows: list[dict] = []
    models = predictions["model"].unique() if "model" in predictions.columns else [None]

    for model in models:
        sub = predictions if model is None else predictions[predictions["model"] == model]
        ic = monthly_rank_ic(sub)
        hit = top_k_hit_rate(sub, top_k=top_k)
        name = model or "model"

        if not ic.empty:
            ic_rows.append(
                {
                    "model": name,
                    "rank_ic_mean": float(ic["rank_ic"].mean()),
                    "rank_ic_std": float(ic["rank_ic"].std()),
                    "icir": float(ic["rank_ic"].mean() / ic["rank_ic"].std()) if ic["rank_ic"].std() > 0 else np.nan,
                    "rank_ic_positive_ratio": float((ic["rank_ic"] > 0).mean()),
                    "n_months": len(ic),
                }
            )

        if not hit.empty:
            hit_rows.append(
                {
                    "model": name,
                    "top_k": top_k,
                    "hit_rate_mean": float(hit["hit_rate"].mean()),
                    "n_months": len(hit),
                }
            )

    return {
        "ic_summary": pd.DataFrame(ic_rows),
        "hit_rate_summary": pd.DataFrame(hit_rows),
    }


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    drawdown = nav / peak - 1
    return float(drawdown.min())
