"""Conditional error analysis: when and why the model fails."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import get_path
from src.ml.metrics import monthly_rank_ic, top_k_hit_rate
from src.ml.metrics import compute_performance_metrics


def _market_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "Date"]].copy()
    if "HS300_VOL" in df.columns:
        out["vol_regime"] = np.where(df["HS300_VOL"] > df["HS300_VOL"].median(), "high_vol", "low_vol")
    if "HS300_TREND_6M" in df.columns:
        out["trend_regime"] = np.where(df["HS300_TREND_6M"] > 0, "bull_6m", "bear_6m")
    if "IND_DISP_1M" in df.columns:
        out["dispersion_regime"] = np.where(
            df["IND_DISP_1M"] > df["IND_DISP_1M"].median(),
            "high_dispersion",
            "low_dispersion",
        )
    if "IND_MOM_1M" in df.columns:
        out["momentum_regime"] = np.where(
            df["IND_MOM_1M"].abs() > df["IND_MOM_1M"].abs().median(),
            "strong_momentum",
            "weak_momentum",
        )
    out["year"] = pd.to_datetime(out["date"]).dt.year
    return out


def analyze_monthly_errors(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str = "lambdarank",
    top_k: int = 5,
) -> pd.DataFrame:
    pred = predictions[predictions["model"] == model_name].copy()
    ic = monthly_rank_ic(pred)
    hit = top_k_hit_rate(pred, top_k=top_k)
    ctx = _market_context(df)

    merged = backtest.merge(ic, on="date", how="left", suffixes=("", "_ic"))
    merged = merged.merge(hit[["date", "hit_rate"]], on="date", how="left")
    merged = merged.merge(ctx, on="date", how="left")

    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "date": row["date"],
                "Date": row.get("Date_x", row.get("Date")),
                "year": row.get("year"),
                "portfolio_return": row.get("portfolio_return"),
                "rank_ic": row.get("rank_ic"),
                "top_k_hit_rate": row.get("hit_rate"),
                "turnover": row.get("turnover"),
                "vol_regime": row.get("vol_regime"),
                "trend_regime": row.get("trend_regime"),
                "dispersion_regime": row.get("dispersion_regime"),
                "momentum_regime": row.get("momentum_regime"),
                "failure_flag": bool(
                    (pd.notna(row.get("rank_ic")) and row.get("rank_ic") < 0)
                    or (pd.notna(row.get("portfolio_return")) and row.get("portfolio_return") < 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_errors_by_condition(monthly_errors: pd.DataFrame) -> pd.DataFrame:
    if monthly_errors.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for col in ["year", "vol_regime", "trend_regime", "dispersion_regime", "momentum_regime"]:
        if col not in monthly_errors.columns:
            continue
        for val, group in monthly_errors.groupby(col):
            if pd.isna(val):
                continue
            perf = compute_performance_metrics(group["portfolio_return"], str(val))
            rows.append(
                {
                    "condition_type": col,
                    "condition_value": val,
                    "n_months": len(group),
                    "mean_rank_ic": float(group["rank_ic"].mean()) if group["rank_ic"].notna().any() else np.nan,
                    "mean_hit_rate": float(group["top_k_hit_rate"].mean())
                    if group["top_k_hit_rate"].notna().any()
                    else np.nan,
                    "failure_rate": float(group["failure_flag"].mean()),
                    "annual_return": perf.get("annual_return"),
                    "sharpe": perf.get("sharpe"),
                }
            )
    return pd.DataFrame(rows)


def diagnose_year(
    monthly_errors: pd.DataFrame,
    year: int,
) -> dict:
    sub = monthly_errors[monthly_errors["year"] == year]
    if sub.empty:
        return {}
    return {
        "year": year,
        "n_months": len(sub),
        "mean_rank_ic": float(sub["rank_ic"].mean()),
        "mean_hit_rate": float(sub["top_k_hit_rate"].mean()),
        "failure_rate": float(sub["failure_flag"].mean()),
        "dominant_vol_regime": sub["vol_regime"].mode().iloc[0] if "vol_regime" in sub else None,
        "dominant_momentum_regime": sub["momentum_regime"].mode().iloc[0] if "momentum_regime" in sub else None,
        "narrative": (
            f"{year}: Rank IC={sub['rank_ic'].mean():.3f}, "
            f"Hit={sub['top_k_hit_rate'].mean():.1%}, "
            f"failure months={sub['failure_flag'].sum()}/{len(sub)}"
        ),
    }


def run_error_analysis(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str = "lambdarank",
    top_k: int = 5,
) -> dict[str, pd.DataFrame | dict]:
    monthly = analyze_monthly_errors(predictions, backtest, df, model_name, top_k)
    summary = summarize_errors_by_condition(monthly)
    year_2023 = diagnose_year(monthly, 2023)

    out_dir = get_path("processed") / "ml" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not monthly.empty:
        monthly.to_csv(out_dir / "error_analysis_monthly.csv", index=False, encoding="utf-8-sig")
    if not summary.empty:
        summary.to_csv(out_dir / "error_analysis_summary.csv", index=False, encoding="utf-8-sig")
    if year_2023:
        pd.DataFrame([year_2023]).to_csv(out_dir / "error_analysis_2023.csv", index=False, encoding="utf-8-sig")

    return {"monthly": monthly, "summary": summary, "year_2023": year_2023}
