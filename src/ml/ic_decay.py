"""IC decay: same 1M model predictions evaluated against 1/2/3M forward returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import get_path
from src.ml.data import get_ml_industries, load_ml_dataset
from src.ml.panel import _compound_industry_return


def _forward_return_at_horizon(
    df: pd.DataFrame,
    date: pd.Timestamp,
    industry: str,
    horizon: int,
) -> float:
    match = df.index[df["date"] == date]
    if match.empty:
        return float("nan")
    pos = int(match[0])
    if pos + horizon >= len(df):
        return float("nan")
    return _compound_industry_return(df, industry, pos + 1, horizon)


def compute_ic_decay(
    predictions: pd.DataFrame,
    df: pd.DataFrame | None = None,
    horizons: list[int] | None = None,
    model_name: str = "lambdarank",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each OOS month, compute Rank IC of y_pred vs actual forward return at horizons 1..H.
    Horizon 1 uses stored y_actual; longer horizons compound from master returns.
    """
    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    df = df.sort_values("date").reset_index(drop=True)
    horizons = horizons or [1, 2, 3]

    pred = predictions[predictions["model"] == model_name].copy()
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()

    monthly_rows: list[dict] = []
    for date, group in pred.groupby("date"):
        for h in horizons:
            actuals = []
            preds = []
            for _, row in group.iterrows():
                if h == 1 and pd.notna(row.get("y_actual")):
                    act = float(row["y_actual"])
                else:
                    act = _forward_return_at_horizon(df, row["date"], row["industry"], h)
                if pd.isna(act):
                    continue
                actuals.append(act)
                preds.append(row["y_pred"])
            if len(actuals) < 3:
                continue
            ic, _ = spearmanr(preds, actuals)
            monthly_rows.append(
                {
                    "date": date,
                    "Date": date.strftime("%Y-%m") if hasattr(date, "strftime") else date,
                    "horizon_months": h,
                    "rank_ic": float(ic) if not np.isnan(ic) else np.nan,
                    "n_industries": len(actuals),
                }
            )

    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return monthly, pd.DataFrame()

    summary = (
        monthly.groupby("horizon_months")["rank_ic"]
        .agg(
            rank_ic_mean="mean",
            rank_ic_std="std",
            rank_ic_median="median",
            icir=lambda s: float(s.mean() / s.std()) if s.std() and s.std() > 0 else np.nan,
            n_months="count",
        )
        .reset_index()
    )
    return monthly, summary


def run_ic_decay_analysis(
    predictions: pd.DataFrame,
    df: pd.DataFrame | None = None,
    model_name: str = "lambdarank",
) -> dict[str, pd.DataFrame]:
    monthly, summary = compute_ic_decay(predictions, df, model_name=model_name)
    out_dir = get_path("processed") / "ml" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not monthly.empty:
        monthly.to_csv(out_dir / "ic_decay_monthly.csv", index=False, encoding="utf-8-sig")
    if not summary.empty:
        summary.to_csv(out_dir / "ic_decay_summary.csv", index=False, encoding="utf-8-sig")
    return {"monthly": monthly, "summary": summary}
