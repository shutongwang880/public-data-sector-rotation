"""ML-based Top-K portfolio with risk control, transaction costs, and turnover."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.ml.benchmark import enrich_backtest_with_benchmarks
from src.ml.metrics import compute_prediction_metrics
from src.ml.risk import (
    construct_portfolio_weights,
    get_risk_config,
    score_with_turnover_penalty,
    select_top_k_scored,
    turnover,
)
from src.ml.metrics import compute_performance_metrics


def get_ml_config() -> dict:
    return load_config().get("ml", {})


def _load_industry_vol_lookup() -> pd.DataFrame | None:
    from src.fetch.industry_specific import load_merged_industry_features

    feats = load_merged_industry_features()
    if feats.empty or "IND_VOL" not in feats.columns:
        return None
    return feats[["date", "industry", "IND_VOL"]].dropna(subset=["IND_VOL"])


def _build_vol_map(
    date: pd.Timestamp,
    industries: list[str],
    vol_lookup: pd.DataFrame | None,
    bench_row: pd.Series,
) -> dict[str, float]:
    vol_map: dict[str, float] = {}
    for ind in industries:
        vol = None
        if vol_lookup is not None:
            match = vol_lookup[(vol_lookup["date"] == date) & (vol_lookup["industry"] == ind)]
            if not match.empty:
                vol = float(match.iloc[0]["IND_VOL"])
        if vol is None or pd.isna(vol):
            r = bench_row.get(ind)
            vol = abs(float(r)) if pd.notna(r) else 1.0
        vol_map[ind] = vol
    return vol_map


def _compound_forward(df: pd.DataFrame, start_date: pd.Timestamp, horizon: int, col: str) -> float:
    """Compound `col` returns over `horizon` months starting at `start_date`."""
    match = df.index[df["date"] == start_date]
    if match.empty:
        return 0.0
    pos = int(match[0])
    prod = 1.0
    for j in range(pos, min(pos + horizon, len(df))):
        r = df.iloc[j][col]
        if pd.isna(r):
            return 0.0
        prod *= 1.0 + float(r)
    return prod - 1.0


def _portfolio_return_from_preds(weights: dict[str, float], group: pd.DataFrame) -> float:
    total = 0.0
    for ind, w in weights.items():
        rows = group[group["industry"] == ind]
        if rows.empty:
            continue
        y = rows.iloc[0].get("y_actual")
        if pd.notna(y):
            total += w * float(y)
    return total


def run_ml_backtest(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    top_k: int | None = None,
    weighting: str | None = None,
    turnover_lambda: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_ml_config()
    risk_cfg = get_risk_config()
    top_k = top_k or cfg.get("top_k", 5)
    one_way_cost = cfg.get("transaction_cost_oneway", 0.0015)
    round_trip = 2 * one_way_cost
    lambda_turnover = (
        turnover_lambda
        if turnover_lambda is not None
        else risk_cfg.get("turnover_penalty_lambda", 0.0)
    )
    weight_method = weighting or risk_cfg.get("default_weighting", "equal")

    pred = predictions[predictions["model"] == model_name].copy().sort_values("date")
    vol_lookup = _load_industry_vol_lookup()
    records: list[dict] = []
    holdings_rows: list[dict] = []
    prev_weights: dict[str, float] = {}

    for date, group in pred.groupby("date"):
        scored = score_with_turnover_penalty(group, prev_weights, lambda_turnover)
        selected = select_top_k_scored(scored, top_k, score_col="score")
        if not selected:
            continue

        bench_row = df[df["date"] == date]
        if bench_row.empty:
            continue
        bench_row = bench_row.iloc[0]

        vol_map = _build_vol_map(date, selected, vol_lookup, bench_row)
        score_map = {
            row["industry"]: float(row["score"])
            for _, row in scored[scored["industry"].isin(selected)].iterrows()
        }
        new_weights = construct_portfolio_weights(
            selected,
            vol_map=vol_map,
            method=weight_method,
            score_map=score_map,
        )
        turn = turnover(prev_weights, new_weights) if prev_weights else 1.0
        tx_cost = turn * round_trip

        horizon = int(group["horizon"].iloc[0]) if "horizon" in group.columns else 1
        gross_ret = _portfolio_return_from_preds(new_weights, group)
        net_ret = gross_ret - tx_cost

        ew_ret = float(group["y_actual"].mean()) if group["y_actual"].notna().any() else 0.0
        bench_ret = _compound_forward(df, date, horizon, "HS300")

        records.append(
            {
                "Date": date.strftime("%Y-%m"),
                "date": date,
                "model": model_name,
                "portfolio_return_gross": gross_ret,
                "portfolio_return": net_ret,
                "transaction_cost": tx_cost,
                "turnover": turn,
                "benchmark_return": bench_ret,
                "equal_weight_return": ew_ret,
                "excess_vs_hs300": net_ret - bench_ret,
                "excess_vs_equal_weight": net_ret - ew_ret,
                "n_holdings": len(selected),
                "holdings": ",".join(selected),
                "weights": ",".join(f"{ind}:{new_weights[ind]:.3f}" for ind in selected),
                "weighting": weight_method,
                "turnover_lambda": lambda_turnover,
                "horizon": horizon,
            }
        )
        holdings_rows.append(
            {
                "Date": date.strftime("%Y-%m"),
                "date": date,
                "model": model_name,
                "holdings": ",".join(selected),
                "turnover": turn,
            }
        )
        prev_weights = new_weights

    bt = pd.DataFrame(records)
    if not bt.empty:
        bt["nav"] = (1 + bt["portfolio_return"]).cumprod()
        bt["nav_gross"] = (1 + bt["portfolio_return_gross"]).cumprod()
        bt["hs300_nav"] = (1 + bt["benchmark_return"]).cumprod()
        bt["ew_nav"] = (1 + bt["equal_weight_return"]).cumprod()
        bt["excess_nav"] = bt["nav"] / bt["hs300_nav"]
        bt = enrich_backtest_with_benchmarks(bt, df)

    return bt, pd.DataFrame(holdings_rows)


def evaluate_ml_strategy(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    weighting: str | None = None,
    turnover_lambda: float | None = None,
    top_k: int | None = None,
) -> dict:
    cfg = get_ml_config()
    top_k = top_k or cfg.get("top_k", 5)
    pred_metrics = compute_prediction_metrics(predictions[predictions["model"] == model_name], top_k=top_k)
    bt, _ = run_ml_backtest(
        predictions,
        df,
        model_name,
        top_k=top_k,
        weighting=weighting,
        turnover_lambda=turnover_lambda,
    )

    perf = {}
    if not bt.empty:
        perf = compute_performance_metrics(bt["portfolio_return"], model_name)
        perf_gross = compute_performance_metrics(bt["portfolio_return_gross"], f"{model_name}_gross")
        perf["annual_return_gross"] = perf_gross.get("annual_return")
        perf["sharpe_gross"] = perf_gross.get("sharpe")
        perf["avg_turnover"] = float(bt["turnover"].mean())
        perf["avg_transaction_cost"] = float(bt["transaction_cost"].mean())
        perf["total_transaction_cost"] = float(bt["transaction_cost"].sum())

    ic_sum = pred_metrics["ic_summary"]
    hit_sum = pred_metrics["hit_rate_summary"]
    row = {"model": model_name, **perf}

    if not ic_sum.empty:
        ic_row = ic_sum[ic_sum["model"] == model_name]
        if not ic_row.empty:
            row.update(
                {
                    "rank_ic_mean": float(ic_row["rank_ic_mean"].iloc[0]),
                    "rank_ic_std": float(ic_row["rank_ic_std"].iloc[0]),
                    "icir": float(ic_row["icir"].iloc[0]),
                    "rank_ic_positive_ratio": float(ic_row["rank_ic_positive_ratio"].iloc[0]),
                    "ic_mean": float(ic_row["rank_ic_mean"].iloc[0]),
                    "ic_positive_ratio": float(ic_row["rank_ic_positive_ratio"].iloc[0]),
                }
            )

    if not hit_sum.empty:
        hit_row = hit_sum[hit_sum["model"] == model_name]
        if not hit_row.empty:
            row["top_k_hit_rate"] = float(hit_row["hit_rate_mean"].iloc[0])

    return {
        "performance": row,
        "backtest": bt,
        **pred_metrics,
    }
