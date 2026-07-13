"""Portfolio construction experiments: weighting schemes × turnover penalty grid."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.ml.strategy import evaluate_ml_strategy, run_ml_backtest
from src.ml.metrics import compute_performance_metrics


def get_portfolio_grid() -> tuple[list[str], list[float]]:
    cfg = load_config().get("ml", {}).get("risk_control", {})
    weightings = cfg.get(
        "portfolio_weightings",
        ["equal", "vol_adjusted", "risk_parity"],
    )
    lambdas = cfg.get("turnover_penalty_grid", [0.0, 0.05, 0.1, 0.2])
    return list(weightings), [float(x) for x in lambdas]


def run_portfolio_experiments(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    experiment_label: str = "regression",
) -> pd.DataFrame:
    weightings, lambdas = get_portfolio_grid()
    rows: list[dict] = []

    for weighting in weightings:
        for lam in lambdas:
            bt, _ = run_ml_backtest(
                predictions,
                df,
                model_name,
                weighting=weighting,
                turnover_lambda=lam,
            )
            if bt.empty:
                continue
            perf = compute_performance_metrics(bt["portfolio_return"], model_name)
            rows.append(
                {
                    "experiment": experiment_label,
                    "model": model_name,
                    "weighting": weighting,
                    "turnover_lambda": lam,
                    "annual_return": perf.get("annual_return"),
                    "sharpe": perf.get("sharpe"),
                    "max_drawdown": perf.get("max_drawdown"),
                    "avg_turnover": float(bt["turnover"].mean()),
                    "avg_transaction_cost": float(bt["transaction_cost"].mean()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out_dir = get_path("processed") / "ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"portfolio_experiments_{experiment_label}.csv"
        out.to_csv(out_dir / fname, index=False, encoding="utf-8-sig")
    return out
