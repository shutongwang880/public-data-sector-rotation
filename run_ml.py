#!/usr/bin/env python3
"""
ML pipeline (research-grade):
  Validation → LambdaRank → Softmax Top-K → Rank IC / IC Decay / Error Analysis

Usage:
    python run_ml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_runtime_deps

ensure_runtime_deps("pandas")

import pandas as pd

from src.config import get_path, load_config
from src.data.feature_selection import count_selected_features
from src.ml.benchmark import (
    run_all_rule_baselines,
    run_random_selection_backtest,
    summarize_all_benchmarks,
)
from src.ml.case_study import run_case_studies
from src.ml.data import get_ml_industries, load_ml_dataset
from src.ml.error_analysis import run_error_analysis
from src.ml.feature_filter import run_feature_selection_pipeline
from src.ml.ic_decay import run_ic_decay_analysis
from src.ml.nested_walkforward import summarize_validation_selection, walk_forward_lambdarank_nested
from src.ml.portfolio_experiments import run_portfolio_experiments
from src.ml.strategy import evaluate_ml_strategy
from src.ml.subperiod import analyze_bull_bear, analyze_subperiods
from src.ml.visualization import (
    generate_ml_figures,
    plot_ic_decay,
    plot_turnover_sweep,
)
from src.ml.walkforward import run_regression_models


def _evaluate_primary(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    weighting: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = evaluate_ml_strategy(predictions, df, model_name, weighting=weighting)
    row = result["performance"]
    row["experiment"] = "ranking"
    row["weighting"] = weighting
    bt = result["backtest"]
    out_dir = get_path("processed") / "ml"
    if not bt.empty:
        bt.to_csv(out_dir / f"backtest_{model_name}.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame([row]), bt


def main() -> None:
    print("=" * 60)
    print("ML Pipeline: Ranking + Nested Validation + Softmax Portfolio")
    print("=" * 60)

    cfg = load_config().get("ml", {})
    df = load_ml_dataset(use_training_sample=True)
    industries = get_ml_industries(df)
    feat_counts = count_selected_features()
    primary = cfg.get("primary_model", "lambdarank")
    weighting = cfg.get("risk_control", {}).get("default_weighting", "softmax")

    print(f"\nSample: {len(df)} months ({df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]})")
    print(f"Industries: {len(industries)}")
    print(f"Curated features: ~{feat_counts['total_approx']}")
    print(f"Primary path: {primary} | Portfolio: {weighting}")

    out_dir = get_path("processed") / "ml"
    fig_dir = get_path("figures") / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/10] Feature selection (Correlation + SHAP + Importance)...")
    fs = run_feature_selection_pipeline(df)
    print(f"  Initial: {len(fs['initial'])} → After corr: {len(fs['after_correlation'])} → Selected: {len(fs['selected'])}")

    print("\n[2/10] Nested Walk-Forward LambdaRank (Train → Val → Test)...")
    print("  (约 10–20 分钟，期间可能无输出；若看到 sklearn 警告可忽略，不是报错)")
    rank_pred, sel_log = walk_forward_lambdarank_nested(df)
    if not rank_pred.empty:
        rank_pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        rank_pred.to_csv(out_dir / "predictions_lambdarank_nested.csv", index=False, encoding="utf-8-sig")
    if not sel_log.empty:
        val_summary = summarize_validation_selection(sel_log)
        val_summary.to_csv(out_dir / "validation" / "selection_summary.csv", index=False, encoding="utf-8-sig")
        print(f"  OOS months: {rank_pred['date'].nunique() if not rank_pred.empty else 0}")
        if "n_estimators" in sel_log.columns:
            print(f"  Most selected n_estimators: {sel_log['n_estimators'].mode().iloc[0]}")

    print("\n[3/10] Primary strategy evaluation (Rank IC + Softmax Top-K)...")
    primary_summary, primary_bt = _evaluate_primary(rank_pred, df, primary, weighting)
    primary_summary.to_csv(out_dir / "model_comparison_primary.csv", index=False, encoding="utf-8-sig")
    if not primary_summary.empty:
        print(
            primary_summary[
                ["model", "annual_return", "sharpe", "rank_ic_mean", "icir", "top_k_hit_rate", "weighting"]
            ].to_string(index=False)
        )

    print("\n[4/10] Regression baselines (comparison only)...")
    reg_pred = run_regression_models(df)
    if not reg_pred.empty:
        reg_pred.to_csv(out_dir / "predictions_regression.csv", index=False, encoding="utf-8-sig")

    print("\n[5/10] Portfolio weighting experiments...")
    if not rank_pred.empty:
        port = run_portfolio_experiments(rank_pred, df, primary, "ranking")
        if not port.empty:
            plot_turnover_sweep(port, fig_dir, "ranking")

    oos_dates = primary_bt["date"] if not primary_bt.empty else None

    print("\n[6/10] Rule-based + random benchmarks...")
    random_bt = run_random_selection_backtest(df, dates=oos_dates)
    if not random_bt.empty:
        random_bt.to_csv(out_dir / "backtest_random_topk.csv", index=False, encoding="utf-8-sig")
    rule_baselines = run_all_rule_baselines(df, dates=oos_dates, weighting=weighting)
    if not primary_bt.empty:
        bench = summarize_all_benchmarks(primary_bt, rule_baselines, random_bt)
        bench.to_csv(out_dir / "benchmark_comparison.csv", index=False, encoding="utf-8-sig")
        print(bench[["benchmark", "annual_return", "sharpe", "max_drawdown"]].to_string(index=False))

    print("\n[7/10] IC Decay analysis...")
    ic_decay = run_ic_decay_analysis(rank_pred, df, model_name=primary)
    if ic_decay.get("summary") is not None and not ic_decay["summary"].empty:
        print(ic_decay["summary"].to_string(index=False))
        plot_ic_decay(ic_decay["summary"], fig_dir)

    print("\n[8/10] Error analysis...")
    if not primary_bt.empty and not rank_pred.empty:
        err = run_error_analysis(rank_pred, primary_bt, df, model_name=primary)
        if err.get("year_2023"):
            print(f"  2023: {err['year_2023'].get('narrative', '')}")

    print("\n[9/10] SHAP case studies...")
    if not primary_bt.empty and not rank_pred.empty:
        cases = run_case_studies(rank_pred, primary_bt, df)
        print(f"  Case study rows: {len(cases)}")

    print("\n[10/10] Sub-period + report...")
    if not primary_bt.empty:
        sub = analyze_subperiods(primary_bt)
        bb = analyze_bull_bear(primary_bt, df)
        if not sub.empty:
            sub.to_csv(out_dir / "subperiod_analysis.csv", index=False, encoding="utf-8-sig")
        if not bb.empty:
            bb.to_csv(out_dir / "bull_bear_analysis.csv", index=False, encoding="utf-8-sig")

    if cfg.get("stability", {}).get("enabled", True) and not rank_pred.empty:
        from src.ml.stability import run_stability_analysis
        from src.ml.visualization import plot_stability_summary

        stab = run_stability_analysis(df, rank_pred, primary)
        if stab.get("summary") is not None and not stab["summary"].empty:
            plot_stability_summary(stab["summary"], fig_dir)

    from src.report.word_export import generate_word

    word_path = generate_word()
    print(f"  Word report: {word_path}")

    backtests = {primary: primary_bt} if not primary_bt.empty else {}
    generate_ml_figures(backtests, pd.DataFrame(), fig_dir, best_model=primary)

    print("\n" + "=" * 60)
    print("Done. Pipeline: Nested Val → LambdaRank → Softmax → Rank IC / IC Decay")
    print("  python run_dashboard.py  — interactive dashboard")
    print("  python generate_word.py  — Word business report")
    print("=" * 60)


if __name__ == "__main__":
    main()
