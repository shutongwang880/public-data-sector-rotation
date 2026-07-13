"""Sub-period and bull/bear conditional backtest analysis."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.ml.metrics import compute_performance_metrics


def _period_slice(bt: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (bt["date"] >= pd.Timestamp(start)) & (bt["date"] < pd.Timestamp(end))
    return bt.loc[mask].copy()


def analyze_subperiods(bt: pd.DataFrame, periods: dict[str, list[str]] | None = None) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()

    cfg = load_config()
    periods = periods or cfg.get("backtest", {}).get(
        "subperiods",
        {
            "2020-2021": ["2020-01-01", "2022-01-01"],
            "2022": ["2022-01-01", "2023-01-01"],
            "2023": ["2023-01-01", "2024-01-01"],
            "2024": ["2024-01-01", "2025-01-01"],
            "2025": ["2025-01-01", "2026-01-01"],
        },
    )

    model = bt["model"].iloc[0] if "model" in bt.columns else "model"
    rows: list[dict] = []
    for label, (start, end) in periods.items():
        sub = _period_slice(bt, start, end)
        if len(sub) < 3:
            continue
        perf = compute_performance_metrics(sub["portfolio_return"], model)
        hs_perf = compute_performance_metrics(sub["benchmark_return"], "HS300") if "benchmark_return" in sub else {}
        rows.append(
            {
                "period": label,
                "n_months": len(sub),
                "annual_return": perf.get("annual_return"),
                "sharpe": perf.get("sharpe"),
                "max_drawdown": perf.get("max_drawdown"),
                "hs300_annual_return": hs_perf.get("annual_return"),
                "excess_vs_hs300": perf.get("annual_return", 0) - hs_perf.get("annual_return", 0)
                if hs_perf
                else None,
            }
        )
    return pd.DataFrame(rows)


def analyze_bull_bear(bt: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Split months by HS300 6M trend sign (risk-on vs risk-off)."""
    if bt.empty or "HS300_TREND_6M" not in df.columns:
        return pd.DataFrame()

    regime = df[["date", "HS300_TREND_6M"]].drop_duplicates("date")
    merged = bt.merge(regime, on="date", how="left")
    merged["regime"] = merged["HS300_TREND_6M"].apply(
        lambda x: "bull_risk_on" if pd.notna(x) and x > 0 else "bear_risk_off"
    )

    model = bt["model"].iloc[0] if "model" in bt.columns else "model"
    rows: list[dict] = []
    for regime_name, sub in merged.groupby("regime"):
        if len(sub) < 3:
            continue
        perf = compute_performance_metrics(sub["portfolio_return"], model)
        rows.append(
            {
                "regime": regime_name,
                "n_months": len(sub),
                "annual_return": perf.get("annual_return"),
                "sharpe": perf.get("sharpe"),
                "max_drawdown": perf.get("max_drawdown"),
            }
        )
    return pd.DataFrame(rows)
