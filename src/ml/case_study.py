"""Case studies: SHAP explanations for specific prediction dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data.feature_catalog import CATEGORY_LABELS, classify_feature_category
from src.ml.data import load_ml_dataset
from src.ml.features import prepare_panel_matrix
from src.ml.interpretability import _friendly_feature_name
from src.ml.models import create_ranking_model
from src.ml.panel import build_long_panel, get_panel_feature_columns


def _pick_case_dates(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    n_dates: int = 3,
) -> list[pd.Timestamp]:
    merged = backtest.merge(
        predictions.groupby("date")["y_pred"].mean().reset_index(),
        on="date",
        how="left",
    )
    if merged.empty:
        return sorted(predictions["date"].unique())[:n_dates]

    best = merged.nlargest(max(1, n_dates // 2 + 1), "portfolio_return")["date"].tolist()
    worst = merged.nsmallest(max(1, n_dates // 2), "portfolio_return")["date"].tolist()
    dates = list(dict.fromkeys(best + worst))[:n_dates]
    return dates


def explain_prediction_date(
    date: pd.Timestamp,
    predictions: pd.DataFrame,
    df: pd.DataFrame | None = None,
    top_industries: int = 1,
    top_features: int = 5,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        return pd.DataFrame()

    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    panel = build_long_panel(df)
    feature_cols = get_panel_feature_columns(panel, include_industry_id=True)

    day_pred = predictions[(predictions["date"] == date) & (predictions["model"] == "lambdarank")]
    if day_pred.empty:
        return pd.DataFrame()

    top_inds = day_pred.nlargest(top_industries, "y_pred")["industry"].tolist()
    train = panel[panel["target_date"] < day_pred["target_date"].iloc[0]]
    if train.empty:
        return pd.DataFrame()

    X_train, cols, state = prepare_panel_matrix(train, feature_cols, "lambdarank", fit=True)
    rel = (
        train.groupby("target_date")["y"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"),
                q=min(10, len(s)),
                labels=False,
                duplicates="drop",
            ).fillna(0)
        )
        .astype(int)
        .clip(0, 9)
        .to_numpy()
    )
    groups = train.groupby("target_date", sort=True).size().to_numpy()
    model = create_ranking_model()
    model.fit(X_train, rel, group=groups)

    rows: list[dict] = []
    for industry in top_inds:
        test_row = panel[(panel["date"] == date) & (panel["industry"] == industry)]
        if test_row.empty:
            test_row = day_pred[day_pred["industry"] == industry]
        if test_row.empty:
            continue

        if "date" in panel.columns:
            feat_row = panel[(panel["date"] == date) & (panel["industry"] == industry)]
        else:
            feat_row = test_row
        if feat_row.empty:
            continue

        X_one, _, _ = prepare_panel_matrix(feat_row, feature_cols, "lambdarank", dummy_state=state, fit=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_one)[0]
        pred_score = float(day_pred[day_pred["industry"] == industry]["y_pred"].iloc[0])

        feat_imp = sorted(
            zip(cols, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_features]

        for rank, (feat, val) in enumerate(feat_imp, start=1):
            cat = classify_feature_category(feat)
            rows.append(
                {
                    "date": date,
                    "Date": date.strftime("%Y-%m") if hasattr(date, "strftime") else date,
                    "industry": industry,
                    "predicted_score": pred_score,
                    "rank_in_portfolio": 1,
                    "feature_rank": rank,
                    "feature": feat,
                    "feature_label": _friendly_feature_name(feat),
                    "category": cat,
                    "category_label": CATEGORY_LABELS.get(cat, cat),
                    "shap_value": float(val),
                    "direction": "positive" if val > 0 else "negative",
                }
            )

    return pd.DataFrame(rows)


def run_case_studies(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    df: pd.DataFrame | None = None,
    dates: list[pd.Timestamp] | None = None,
    n_dates: int = 3,
) -> pd.DataFrame:
    cfg = load_config().get("ml", {}).get("case_study", {})
    n_dates = cfg.get("n_dates", n_dates)

    if dates is None:
        dates = _pick_case_dates(predictions, backtest, n_dates)

    frames = []
    for d in dates:
        part = explain_prediction_date(d, predictions, df)
        if not part.empty:
            frames.append(part)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_dir = get_path("processed") / "ml" / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out.empty:
        out.to_csv(out_dir / "case_study_shap.csv", index=False, encoding="utf-8-sig")
    return out
