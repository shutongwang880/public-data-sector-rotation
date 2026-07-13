"""Walk-forward training and prediction on long-format panel data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.data import get_ml_config, load_ml_dataset
from src.ml.features import prepare_panel_matrix
from src.ml.models import create_ranking_model, create_regression_model
from src.ml.panel import build_long_panel, get_panel_feature_columns


def _select_train_panel(
    panel: pd.DataFrame,
    target_date: pd.Timestamp,
    train_mode: str,
    rolling_window: int | None,
    min_train_months: int,
) -> pd.DataFrame:
    all_dates = sorted(panel["target_date"].unique())
    past_dates = [d for d in all_dates if d < target_date]

    if train_mode == "rolling" and rolling_window:
        if len(past_dates) < rolling_window:
            return pd.DataFrame()
        cutoff = past_dates[-rolling_window]
        return panel[(panel["target_date"] >= cutoff) & (panel["target_date"] < target_date)]

    train = panel[panel["target_date"] < target_date]
    if len(past_dates) < min_train_months:
        return pd.DataFrame()
    return train


def walk_forward_panel_regression(
    df: pd.DataFrame,
    model_name: str,
    min_train_months: int | None = None,
    feature_cols: list[str] | None = None,
    horizon: int = 1,
    train_mode: str = "expanding",
    rolling_window: int | None = None,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pooled panel walk-forward with expanding or rolling training windows."""
    cfg = get_ml_config()
    min_train_months = min_train_months or cfg.get("min_train_months", 36)

    if panel is None:
        panel = build_long_panel(df, horizon=horizon)
    if panel.empty:
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = get_panel_feature_columns(panel)
    else:
        feature_cols = [c for c in feature_cols if c in panel.columns]
    if not feature_cols:
        return pd.DataFrame()

    forecast_dates = sorted(panel["target_date"].unique())
    records: list[dict] = []

    for target_date in forecast_dates:
        train = _select_train_panel(panel, target_date, train_mode, rolling_window, min_train_months)
        test = panel[panel["target_date"] == target_date]
        if train.empty or test.empty:
            continue

        X_train, _, state = prepare_panel_matrix(train, feature_cols, model_name, fit=True)
        y_train = train["y"].to_numpy(dtype=float)
        X_test, _, _ = prepare_panel_matrix(test, feature_cols, model_name, dummy_state=state, fit=False)

        if model_name in {"ridge", "lasso"} and (np.isnan(X_train).any() or np.isnan(X_test).any()):
            continue

        model = create_regression_model(model_name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for (_, row), pred in zip(test.iterrows(), preds):
            records.append(
                {
                    "Date": row["date"].strftime("%Y-%m") if hasattr(row["date"], "strftime") else row["date"],
                    "date": row["date"],
                    "feature_date": row["feature_date"],
                    "target_date": row["target_date"],
                    "horizon": int(row.get("horizon", horizon)),
                    "industry": row["industry"],
                    "model": model_name,
                    "y_pred": float(pred),
                    "y_actual": float(row["y"]),
                    "train_mode": train_mode,
                    "rolling_window": rolling_window if train_mode == "rolling" else None,
                }
            )

    return pd.DataFrame(records)


def walk_forward_lambdarank(
    df: pd.DataFrame,
    min_train_months: int | None = None,
    horizon: int = 1,
    train_mode: str = "expanding",
    rolling_window: int | None = None,
) -> pd.DataFrame:
    cfg = get_ml_config()
    min_train_months = min_train_months or cfg.get("min_train_months", 36)

    panel = build_long_panel(df, horizon=horizon)
    if panel.empty:
        return pd.DataFrame()

    feature_cols = get_panel_feature_columns(panel, include_industry_id=True)
    forecast_dates = sorted(panel["target_date"].unique())
    records: list[dict] = []

    for target_date in forecast_dates:
        train = _select_train_panel(panel, target_date, train_mode, rolling_window, min_train_months)
        test = panel[panel["target_date"] == target_date]
        if train.empty or len(test) < 3:
            continue

        X_train, _, state = prepare_panel_matrix(train, feature_cols, "lambdarank", fit=True)
        X_test, _, _ = prepare_panel_matrix(test, feature_cols, "lambdarank", dummy_state=state, fit=False)

        rel_train = (
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
        groups_train = train.groupby("target_date", sort=True).size().to_numpy()

        model = create_ranking_model()
        model.fit(X_train, rel_train, group=groups_train)
        scores = model.predict(X_test)

        for (_, row), score in zip(test.iterrows(), scores):
            records.append(
                {
                    "Date": row["date"].strftime("%Y-%m") if hasattr(row["date"], "strftime") else row["date"],
                    "date": row["date"],
                    "feature_date": row["feature_date"],
                    "target_date": row["target_date"],
                    "horizon": int(row.get("horizon", horizon)),
                    "industry": row["industry"],
                    "model": "lambdarank",
                    "y_pred": float(score),
                    "y_actual": float(row["y"]),
                    "train_mode": train_mode,
                    "rolling_window": rolling_window if train_mode == "rolling" else None,
                }
            )

    return pd.DataFrame(records)


def run_regression_models(df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
    cfg = get_ml_config()
    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    names = cfg.get("regression_models", ["ridge", "lasso", "random_forest", "lightgbm", "xgboost"])
    frames = []
    for name in names:
        pred = walk_forward_panel_regression(df, name, **kwargs)
        if not pred.empty:
            frames.append(pred)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_ranking_models(df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    pred = walk_forward_lambdarank(df, **kwargs)
    return pred


def run_all_models(df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
    reg = run_regression_models(df, **kwargs)
    rank = run_ranking_models(df, **kwargs)
    frames = [f for f in [reg, rank] if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
