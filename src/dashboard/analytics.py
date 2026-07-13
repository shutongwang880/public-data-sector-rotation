"""Business analytics data layer for the Decision Support Dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.dashboard.data_loader import (
    load_backtest,
    load_benchmarks,
    load_case_studies,
    load_ic_decay,
    load_live_forecast,
    load_predictions,
    load_primary_metrics,
)
from src.ml.risk import construct_portfolio_weights

# ── Indicator metadata (business-facing labels) ──────────────────────────────

INDICATOR_META: dict[str, dict] = {
    "PMI": {"label": "Manufacturing PMI", "unit": "index", "module": "Macroeconomy"},
    "CPI": {"label": "CPI YoY", "unit": "%", "module": "Macroeconomy"},
    "PPI": {"label": "PPI YoY", "unit": "%", "module": "Macroeconomy"},
    "M2": {"label": "M2 Growth", "unit": "%", "module": "Macroeconomy"},
    "TSF": {"label": "Total Social Financing", "unit": "%", "module": "Macroeconomy"},
    "MARGIN_BAL": {"label": "Margin Balance", "unit": "CNY bn", "module": "Liquidity"},
    "NORTH_FLOW": {"label": "Northbound Net Flow", "unit": "CNY bn", "module": "Liquidity"},
    "TURNOVER": {"label": "Market Turnover", "unit": "CNY", "module": "Market Sentiment"},
    "HS300_VOL": {"label": "CSI 300 Volatility", "unit": "", "module": "Market Risk"},
    "MARKET_BREADTH": {"label": "Market Breadth", "unit": "", "module": "Market Sentiment"},
    "HS300_TREND_3M": {"label": "CSI 300 3M Trend", "unit": "", "module": "Market Risk"},
    "IND_MOM_1M": {"label": "Sector 1M Momentum", "unit": "", "module": "Sector Aggregate"},
    "IND_DISP_1M": {"label": "Cross-Section Dispersion", "unit": "", "module": "Sector Aggregate"},
}

FEATURE_LABELS: dict[str, str] = {
    "IND_REL_MOM_12M": "12M Relative Momentum",
    "IND_PX_MOM_12M": "12M Price Momentum",
    "IND_PX_MOM_3M": "3M Price Momentum",
    "IND_REL_STRENGTH": "Relative Strength",
    "IND_VOL": "Sector Volatility",
    "IND_MOM_CS_PCT": "Cross-Section Momentum Percentile",
    "MARGIN_BAL_zscore": "Margin Balance (z-score)",
    "PMI_zscore": "PMI (z-score)",
    "NORTH_FLOW_zscore": "Northbound Flow (z-score)",
    "TURNOVER_zscore": "Market Turnover (z-score)",
    "HS300_VOL_zscore": "Market Volatility (z-score)",
    "TSF_zscore": "Total Social Financing (z-score)",
    "M2_zscore": "M2 Growth (z-score)",
    "MARGIN_CHG_zscore": "Margin Balance Change (z-score)",
}

WEIGHTING_LABELS = {
    "equal": "Equal Weight",
    "softmax": "Score-Weighted (Softmax)",
    "vol_adjusted": "Volatility Adjusted",
    "risk_parity": "Risk Parity",
}


def _read_csv(rel: str) -> pd.DataFrame | None:
    path = get_path("processed") / rel
    if not path.exists():
        return None
    cols = pd.read_csv(path, nrows=0).columns
    parse = [c for c in cols if "date" in c.lower() or c in ("feature_date", "target_date")]
    return pd.read_csv(path, parse_dates=parse if parse else None)


def load_features_timeseries() -> pd.DataFrame:
    path = get_path("processed") / "features_standardized.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def infer_market_regime(row: pd.Series) -> dict:
    """Classify current market state for executive summary."""
    pmi_z = float(row.get("PMI_zscore", 0) or 0)
    margin_z = float(row.get("MARGIN_BAL_zscore", 0) or 0)
    vol_z = float(row.get("HS300_VOL_zscore", 0) or 0)
    north_z = float(row.get("NORTH_FLOW_zscore", 0) or 0)
    mom_z = float(row.get("IND_MOM_1M_zscore", 0) or 0)

    if margin_z > 0.3 and north_z > 0:
        liquidity = "Accommodative Liquidity"
    elif margin_z < -0.3:
        liquidity = "Tight Liquidity"
    else:
        liquidity = "Neutral Liquidity"

    if vol_z > 0.5:
        risk = "Elevated Risk"
    elif vol_z < -0.3:
        risk = "Low Risk"
    else:
        risk = "Neutral Risk"

    if pmi_z > 0.2:
        growth = "Expansion"
    elif pmi_z < -0.2:
        growth = "Soft Growth"
    else:
        growth = "Stable Growth"

    if mom_z > 0.3:
        momentum = "Strong Sector Momentum"
    elif mom_z < -0.3:
        momentum = "Weak Sector Momentum"
    else:
        momentum = "Neutral Sector Momentum"

    return {
        "liquidity": liquidity,
        "risk": risk,
        "growth": growth,
        "momentum": momentum,
        "headline_regime": f"{liquidity} · {risk} · {growth}",
    }


def load_market_environment() -> dict:
    """Market Environment page data."""
    feat = load_features_timeseries()
    if feat.empty:
        return {}

    latest = feat.iloc[-1]
    regime = infer_market_regime(latest)

    modules: dict[str, list] = {}
    for col, meta in INDICATOR_META.items():
        mod = meta["module"]
        zcol = f"{col}_zscore"
        entry = {
            "indicator": col,
            "label": meta["label"],
            "value": latest.get(col),
            "zscore": latest.get(zcol) if zcol in feat.columns else None,
            "unit": meta["unit"],
        }
        if mod not in modules:
            modules[mod] = []
        modules[mod].append(entry)

    history = feat.tail(24)
    as_of = pd.Timestamp(latest["date"]).strftime("%b %Y")
    return {
        "as_of": as_of,
        "regime": regime,
        "modules": modules,
        "history": history,
    }


def _industry_vol_map(industries: list[str]) -> dict[str, float]:
    raw = get_path("raw") / "industry_monthly_returns.csv"
    if not raw.exists():
        return {}
    ret = pd.read_csv(raw, parse_dates=["date"])
    vol_map: dict[str, float] = {}
    for ind in industries:
        if ind not in ret.columns:
            continue
        s = ret[ind].dropna().tail(12)
        vol_map[ind] = float(s.std()) if len(s) >= 3 else 0.05
    return vol_map


def load_portfolio_schemes(forecast: dict | None = None) -> list[dict]:
    """Three allocation schemes for Top-K industries."""
    fc = forecast or load_live_forecast()
    if not fc:
        return []

    top_inds = [h["industry"] for h in fc.get("top_holdings", [])]
    score_map = {h["industry"]: h["score"] for h in fc.get("top_holdings", [])}
    vol_map = _industry_vol_map(top_inds)
    cfg = load_config().get("ml", {}).get("risk_control", {})
    max_w = float(cfg.get("max_industry_weight", 0.20))

    schemes = []
    for method in ("equal", "vol_adjusted", "risk_parity"):
        weights = construct_portfolio_weights(
            top_inds,
            vol_map=vol_map,
            method=method,
            max_weight=max_w,
            score_map=score_map if method == "softmax" else None,
        )
        schemes.append(
            {
                "method": method,
                "label": WEIGHTING_LABELS.get(method, method),
                "weights": [{"industry": k, "weight": v} for k, v in weights.items()],
            }
        )

    softmax_weights = [
        {"industry": h["industry"], "weight": h["weight"]}
        for h in fc.get("top_holdings", [])
        if h.get("weight") is not None
    ]
    schemes.insert(
        0,
        {
            "method": "softmax",
            "label": WEIGHTING_LABELS["softmax"],
            "weights": softmax_weights,
            "primary": True,
        },
    )
    return schemes


def load_global_shap() -> dict:
    shap_path = get_path("processed") / "ml" / "feature_selection" / "shap_importance.csv"
    cat_path = get_path("processed") / "ml" / "interpretability" / "category_importance_xgboost.csv"
    out: dict = {"features": [], "categories": []}

    if shap_path.exists():
        df = pd.read_csv(shap_path).head(12)
        out["features"] = [
            {
                "feature": r["feature"],
                "label": FEATURE_LABELS.get(r["feature"], r["feature"]),
                "importance": float(r["mean_abs_shap"]),
            }
            for _, r in df.iterrows()
        ]

    if cat_path.exists():
        df = pd.read_csv(cat_path)
        out["categories"] = [
            {
                "category": r["category_label"],
                "share": float(r["share"]),
                "importance": float(r["mean_abs_shap"]),
            }
            for _, r in df.iterrows()
        ]
    return out


def load_local_shap(industry: str) -> list[dict]:
    cases = load_case_studies()
    if cases.empty:
        return []
    sub = cases[cases["industry"] == industry].sort_values("feature_rank")
    if sub.empty:
        latest = load_case_studies()
        sub = latest[latest["industry"] == industry] if not latest.empty else sub
    return [
        {
            "feature": r.get("feature_label", r.get("feature")),
            "category": r.get("category_label", ""),
            "shap": float(r["shap_value"]),
            "direction": r.get("direction", ""),
        }
        for _, r in sub.head(8).iterrows()
    ]


def generate_decision_summary(
    forecast: dict | None = None,
    market: dict | None = None,
    shap: dict | None = None,
) -> str:
    fc = forecast or load_live_forecast()
    mk = market or load_market_environment()
    sh = shap or load_global_shap()

    if not fc:
        return "No forecast available. Run the data pipeline and reload the dashboard."

    top_names = ", ".join(h["industry"] for h in fc.get("top_holdings", [])[:3])
    regime = mk.get("regime", {}) if mk else {}
    drivers: list[str] = []
    if sh.get("features"):
        drivers = [f["label"] for f in sh["features"][:3]]
    elif sh.get("categories"):
        drivers = [c["category"] for c in sh["categories"][:3]]

    driver_text = ", ".join(drivers) if drivers else "sector momentum and market liquidity"
    regime_text = regime.get("headline_regime", "the prevailing market regime")

    return (
        f"<b>Decision Summary.</b> For {fc.get('forecast_month', 'the next month')}, "
        f"the model assigns the highest relative-return potential to {top_names}. "
        f"The prevailing environment is characterised by {regime_text}. "
        f"Ranking is primarily driven by {driver_text}. "
        f"A score-weighted (Softmax) allocation is recommended, subject to monthly review."
    )


def load_sector_intelligence(industry: str) -> dict:
    pred = load_predictions()
    if pred.empty:
        return {}

    if "model" in pred.columns:
        pred = pred[pred["model"] == load_config().get("ml", {}).get("primary_model", "lambdarank")]

    ind_pred = pred[pred["industry"] == industry].sort_values("date").tail(12)
    latest_score = float(ind_pred.iloc[-1]["y_pred"]) if not ind_pred.empty else None
    latest_actual = (
        float(ind_pred.iloc[-1]["y_actual"])
        if not ind_pred.empty and pd.notna(ind_pred.iloc[-1].get("y_actual"))
        else None
    )

    all_latest = pred[pred["date"] == pred["date"].max()].sort_values("y_pred", ascending=False).reset_index(drop=True)
    match = all_latest[all_latest["industry"] == industry]
    rank = int(match.index[0]) + 1 if not match.empty else None

    shap_local = load_local_shap(industry)
    explanation = _explain_sector(industry, rank, latest_score, shap_local)

    return {
        "industry": industry,
        "rank": rank,
        "latest_score": latest_score,
        "latest_actual": latest_actual,
        "history": ind_pred[["date", "y_pred", "y_actual"]].to_dict(orient="records") if not ind_pred.empty else [],
        "shap_drivers": shap_local,
        "explanation": explanation,
    }


def _explain_sector(
    industry: str,
    rank: int | None,
    score: float | None,
    shap_local: list[dict],
) -> str:
    if not shap_local:
        trend = "remains in the neutral band"
        if rank and rank <= 5:
            trend = "is in the overweight band"
        elif rank and rank >= 27:
            trend = "is in the underweight band"
        return (
            f"{industry} {trend}. Position sizing should be reviewed against "
            f"the current market regime and portfolio constraints."
        )

    pos = [d for d in shap_local if d.get("direction") == "positive"][:2]
    neg = [d for d in shap_local if d.get("direction") == "negative"][:1]
    pos_text = ", ".join(d["feature"] for d in pos) if pos else "momentum factors"
    neg_text = f", partially offset by {neg[0]['feature']}" if neg else ""

    direction = "improving" if rank and rank <= 10 else "softening" if rank and rank >= 22 else "stable"
    return (
        f"{industry} ranks #{rank or '—'} with a {direction} profile. "
        f"The score is supported by {pos_text}{neg_text}."
    )


def load_model_evaluation() -> pd.DataFrame:
    primary = _read_csv("ml/model_comparison_primary.csv")
    comp = _read_csv("ml/model_comparison.csv")
    ranking = _read_csv("ml/model_comparison_ranking.csv")

    frames = []
    if comp is not None and not comp.empty:
        frames.append(comp)
    if ranking is not None and not ranking.empty:
        frames.append(ranking[~ranking["model"].isin(comp["model"] if comp is not None else [])])
    if primary is not None and not primary.empty:
        frames.append(primary)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model"], keep="last")
    col_map = {
        "model": "Model",
        "annual_return": "Annual Return",
        "sharpe": "Sharpe",
        "rank_ic_mean": "Rank IC",
        "top_k_hit_rate": "Top-K Hit Rate",
        "max_drawdown": "Max Drawdown",
    }
    show_cols = [c for c in col_map if c in df.columns]
    out = df[show_cols].rename(columns=col_map)
    return out.sort_values("Sharpe", ascending=False) if "Sharpe" in out.columns else out


def load_portfolio_experiments() -> pd.DataFrame:
    df = _read_csv("ml/portfolio_experiments_ranking.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df[df["turnover_lambda"] == 0].copy()
    sub["weighting"] = sub["weighting"].map(WEIGHTING_LABELS).fillna(sub["weighting"])
    return sub[["weighting", "annual_return", "sharpe", "max_drawdown", "avg_turnover"]]


def load_topk_stability() -> pd.DataFrame:
    path = get_path("processed") / "ml" / "stability" / "stability_top_k.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_pipeline_steps() -> list[dict]:
    return [
        {
            "step": 1,
            "name": "Data Acquisition",
            "label": "Public data ingestion",
            "detail": "AkShare: Shenwan L1 returns, macro, liquidity, market state",
            "output": "data/raw/",
        },
        {
            "step": 2,
            "name": "Data Cleaning",
            "label": "Alignment & quality control",
            "detail": "Missing-value handling, frequency alignment, comparable universe filter",
            "output": "features_clean.csv",
        },
        {
            "step": 3,
            "name": "Feature Engineering",
            "label": "Factor construction",
            "detail": "Macro / liquidity / market / sector factors with publication lags",
            "output": "features_standardized.csv",
        },
        {
            "step": 4,
            "name": "Model Training",
            "label": "Nested walk-forward",
            "detail": "12-month validation window; LambdaRank hyperparameter selection",
            "output": "predictions_lambdarank_nested.csv",
        },
        {
            "step": 5,
            "name": "Prediction",
            "label": "Cross-sectional ranking",
            "detail": "27 comparable sectors ranked for next-month relative performance",
            "output": "predictions_*.csv",
        },
        {
            "step": 6,
            "name": "Portfolio Construction",
            "label": "Top-K weighting",
            "detail": "Equal / vol-adjusted / risk parity / softmax schemes",
            "output": "backtest_lambdarank.csv",
        },
        {
            "step": 7,
            "name": "Evaluation",
            "label": "Performance analytics",
            "detail": "Rank IC, benchmark comparison, IC decay, error analysis",
            "output": "benchmark_comparison.csv",
        },
        {
            "step": 8,
            "name": "Decision Dashboard",
            "label": "Decision support UI",
            "detail": "Executive Overview → Sector Intelligence → Explainable AI",
            "output": "Streamlit Dashboard",
        },
    ]
