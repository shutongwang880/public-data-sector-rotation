"""Benchmark enrichment, random null model, and comparison tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.ml.risk import construct_portfolio_weights, portfolio_return, turnover
from src.ml.metrics import compute_performance_metrics


def load_benchmark_returns() -> pd.DataFrame:
    raw = get_path("raw")
    hs300 = pd.read_csv(raw / "hs300_monthly_returns.csv", parse_dates=["date"])
    out = hs300.rename(columns={"HS300": "benchmark_hs300"})

    csi_path = raw / "csi500_monthly_returns.csv"
    if csi_path.exists():
        csi = pd.read_csv(csi_path, parse_dates=["date"])
        out = out.merge(csi.rename(columns={"CSI500": "benchmark_csi500"}), on="date", how="left")
    return out


def enrich_backtest_with_benchmarks(bt: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt

    out = bt.copy()
    bench = load_benchmark_returns()
    merged = out.merge(bench, on="date", how="left")

    if "benchmark_csi500" in merged.columns:
        merged["csi500_return"] = merged["benchmark_csi500"]
        merged["csi500_nav"] = (1 + merged["csi500_return"].fillna(0)).cumprod()
        merged["excess_vs_csi500"] = merged["portfolio_return"] - merged["csi500_return"]

    if "equal_weight_return" in merged.columns:
        merged["industry_ew_return"] = merged["equal_weight_return"]
        merged["industry_ew_nav"] = merged.get("ew_nav", (1 + merged["industry_ew_return"]).cumprod())

    return merged


def run_random_selection_backtest(
    df: pd.DataFrame,
    dates: pd.Series | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Null model: random Top-K industries each month (equal weight)."""
    cfg = load_config().get("ml", {})
    top_k = top_k or cfg.get("top_k", 5)
    seed = seed if seed is not None else cfg.get("random_state", 42)
    one_way_cost = cfg.get("transaction_cost_oneway", 0.0015)
    round_trip = 2 * one_way_cost

    from src.ml.data import get_ml_industries

    industries = get_ml_industries(df)
    if not industries:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    if dates is None:
        dates = df["date"]

    records: list[dict] = []
    prev_weights: dict[str, float] = {}

    for date in dates:
        bench_row = df[df["date"] == date]
        if bench_row.empty:
            continue
        bench_row = bench_row.iloc[0]
        valid_inds = [c for c in industries if pd.notna(bench_row.get(c))]
        if len(valid_inds) < top_k:
            continue

        selected = list(rng.choice(valid_inds, size=top_k, replace=False))
        new_weights = construct_portfolio_weights(selected, method="equal")
        turn = turnover(prev_weights, new_weights) if prev_weights else 1.0
        tx_cost = turn * round_trip
        gross_ret = portfolio_return(new_weights, bench_row)
        net_ret = gross_ret - tx_cost
        ew_ret = float(bench_row[valid_inds].mean())
        bench_ret = float(bench_row["HS300"])

        records.append(
            {
                "Date": date.strftime("%Y-%m"),
                "date": date,
                "model": "random_topk",
                "portfolio_return_gross": gross_ret,
                "portfolio_return": net_ret,
                "transaction_cost": tx_cost,
                "turnover": turn,
                "benchmark_return": bench_ret,
                "equal_weight_return": ew_ret,
                "n_holdings": len(selected),
            }
        )
        prev_weights = new_weights

    bt = pd.DataFrame(records)
    if not bt.empty:
        bt["nav"] = (1 + bt["portfolio_return"]).cumprod()
        bt["hs300_nav"] = (1 + bt["benchmark_return"]).cumprod()
        bt["ew_nav"] = (1 + bt["equal_weight_return"]).cumprod()
    return bt


def summarize_vs_benchmarks(
    bt: pd.DataFrame,
    random_bt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()

    rows = []
    model = bt["model"].iloc[0] if "model" in bt.columns else "strategy"
    strat = compute_performance_metrics(bt["portfolio_return"], model)
    rows.append({"benchmark": "Strategy (net)", **strat})

    for col, label in [
        ("benchmark_return", "沪深300"),
        ("csi500_return", "中证500"),
        ("equal_weight_return", "行业等权"),
    ]:
        if col not in bt.columns or bt[col].isna().all():
            continue
        perf = compute_performance_metrics(bt[col].dropna(), label)
        rows.append({"benchmark": label, **perf})

    if random_bt is not None and not random_bt.empty:
        perf = compute_performance_metrics(random_bt["portfolio_return"], "随机Top-K")
        rows.append({"benchmark": "随机Top-K（Null）", **perf})

    return pd.DataFrame(rows)


CYCLICAL_INDUSTRIES = [
    "钢铁",
    "有色金属",
    "煤炭",
    "基础化工",
    "建筑材料",
    "机械设备",
    "电力设备",
    "国防军工",
]
DEFENSIVE_INDUSTRIES = [
    "食品饮料",
    "医药生物",
    "公用事业",
    "银行",
    "农林牧渔",
]


def _compound_past_return(series: pd.Series, end_idx: int, window: int) -> float:
    start = max(0, end_idx - window)
    chunk = series.iloc[start:end_idx]
    if chunk.isna().any() or len(chunk) < max(2, window // 2):
        return float("nan")
    return float((1 + chunk).prod() - 1)


def _run_rule_backtest(
    df: pd.DataFrame,
    dates: pd.Series,
    select_fn,
    label: str,
    top_k: int | None = None,
    weighting: str = "equal",
) -> pd.DataFrame:
    cfg = load_config().get("ml", {})
    top_k = top_k or cfg.get("top_k", 5)
    one_way_cost = cfg.get("transaction_cost_oneway", 0.0015)
    round_trip = 2 * one_way_cost

    from src.ml.data import get_ml_industries

    industries = get_ml_industries(df)
    df = df.sort_values("date").reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(df["date"])}

    records: list[dict] = []
    prev_weights: dict[str, float] = {}

    for date in dates:
        if date not in date_to_idx:
            continue
        idx = date_to_idx[date]
        if idx < 2:
            continue
        bench_row = df.iloc[idx]
        valid_inds = [c for c in industries if pd.notna(bench_row.get(c))]
        selected = select_fn(df, idx, valid_inds, top_k)
        if not selected:
            continue

        score_map = {ind: float(i + 1) for i, ind in enumerate(reversed(selected))}
        new_weights = construct_portfolio_weights(
            selected,
            method=weighting,
            score_map=score_map if weighting == "softmax" else None,
        )
        turn = turnover(prev_weights, new_weights) if prev_weights else 1.0
        tx_cost = turn * round_trip
        gross_ret = portfolio_return(new_weights, bench_row)
        net_ret = gross_ret - tx_cost

        records.append(
            {
                "Date": date.strftime("%Y-%m"),
                "date": date,
                "model": label,
                "portfolio_return_gross": gross_ret,
                "portfolio_return": net_ret,
                "transaction_cost": tx_cost,
                "turnover": turn,
                "benchmark_return": float(bench_row.get("HS300", 0)),
                "equal_weight_return": float(bench_row[valid_inds].mean()),
                "n_holdings": len(selected),
                "holdings": ",".join(selected),
            }
        )
        prev_weights = new_weights

    bt = pd.DataFrame(records)
    if not bt.empty:
        bt["nav"] = (1 + bt["portfolio_return"]).cumprod()
    return bt


def run_momentum_topk_backtest(
    df: pd.DataFrame,
    dates: pd.Series | None = None,
    lookback: int = 12,
    top_k: int | None = None,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Baseline: buy top-K industries by trailing 12M momentum."""

    def _select(df_in: pd.DataFrame, idx: int, valid: list[str], k: int) -> list[str]:
        scores = {}
        for ind in valid:
            mom = _compound_past_return(df_in[ind], idx, lookback)
            if pd.notna(mom):
                scores[ind] = mom
        if not scores:
            return []
        return sorted(scores, key=scores.get, reverse=True)[:k]

    use_dates = dates if dates is not None else df["date"]
    return _run_rule_backtest(df, use_dates, _select, "momentum_topk", top_k, weighting)


def run_mean_reversion_backtest(
    df: pd.DataFrame,
    dates: pd.Series | None = None,
    lookback: int = 12,
    top_k: int | None = None,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Baseline: buy bottom-K industries by trailing 12M return (reversal)."""

    def _select(df_in: pd.DataFrame, idx: int, valid: list[str], k: int) -> list[str]:
        scores = {}
        for ind in valid:
            mom = _compound_past_return(df_in[ind], idx, lookback)
            if pd.notna(mom):
                scores[ind] = mom
        if not scores:
            return []
        return sorted(scores, key=scores.get)[:k]

    use_dates = dates if dates is not None else df["date"]
    return _run_rule_backtest(df, use_dates, _select, "mean_reversion", top_k, weighting)


def run_macro_rule_backtest(
    df: pd.DataFrame,
    dates: pd.Series | None = None,
    top_k: int | None = None,
    pmi_threshold: float = 50.0,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Baseline: PMI expansion → cyclicals; PMI contraction → defensives; pick by 3M momentum."""

    def _select(df_in: pd.DataFrame, idx: int, valid: list[str], k: int) -> list[str]:
        pmi = df_in.iloc[idx - 1].get("PMI") if idx > 0 else df_in.iloc[idx].get("PMI")
        if pd.isna(pmi):
            universe = valid
        elif float(pmi) >= pmi_threshold:
            universe = [i for i in valid if i in CYCLICAL_INDUSTRIES]
        else:
            universe = [i for i in valid if i in DEFENSIVE_INDUSTRIES]
        if len(universe) < k:
            universe = valid
        scores = {}
        for ind in universe:
            mom = _compound_past_return(df_in[ind], idx, 3)
            if pd.notna(mom):
                scores[ind] = mom
        if not scores:
            return []
        return sorted(scores, key=scores.get, reverse=True)[:k]

    use_dates = dates if dates is not None else df["date"]
    return _run_rule_backtest(df, use_dates, _select, "macro_rule", top_k, weighting)


def run_all_rule_baselines(
    df: pd.DataFrame,
    dates: pd.Series | None = None,
    weighting: str = "equal",
) -> dict[str, pd.DataFrame]:
    out = {
        "momentum_topk": run_momentum_topk_backtest(df, dates, weighting=weighting),
        "mean_reversion": run_mean_reversion_backtest(df, dates, weighting=weighting),
        "macro_rule": run_macro_rule_backtest(df, dates, weighting=weighting),
    }
    out_dir = get_path("processed") / "ml" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, bt in out.items():
        if not bt.empty:
            bt.to_csv(out_dir / f"backtest_{name}.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_all_benchmarks(
    strategy_bt: pd.DataFrame,
    rule_baselines: dict[str, pd.DataFrame],
    random_bt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    if not strategy_bt.empty:
        model = strategy_bt["model"].iloc[0] if "model" in strategy_bt.columns else "strategy"
        rows.append({"benchmark": "Strategy (ML)", **compute_performance_metrics(strategy_bt["portfolio_return"], model)})

    for col, label in [
        ("benchmark_return", "沪深300"),
        ("equal_weight_return", "行业等权"),
    ]:
        if col in strategy_bt.columns and strategy_bt[col].notna().any():
            rows.append({"benchmark": label, **compute_performance_metrics(strategy_bt[col].dropna(), label)})

    label_map = {
        "momentum_topk": "Momentum Top-5 (12M)",
        "mean_reversion": "Mean Reversion (12M)",
        "macro_rule": "Macro Rule (PMI)",
    }
    for key, bt in rule_baselines.items():
        if not bt.empty:
            rows.append(
                {"benchmark": label_map.get(key, key), **compute_performance_metrics(bt["portfolio_return"], key)}
            )

    if random_bt is not None and not random_bt.empty:
        rows.append({"benchmark": "随机Top-K（Null）", **compute_performance_metrics(random_bt["portfolio_return"], "random")})

    return pd.DataFrame(rows)
