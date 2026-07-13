"""Nested walk-forward: inner validation for hyperparameter selection, outer OOS test."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.ml.data import get_ml_config, load_ml_dataset
from src.ml.features import prepare_panel_matrix
from src.ml.metrics import monthly_rank_ic
from src.ml.models import create_ranking_model, get_rank_param_grid
from src.ml.panel import build_long_panel, get_panel_feature_columns
from src.ml.walkforward import _select_train_panel



def _compute_relevance(train: pd.DataFrame) -> np.ndarray:
    return (
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


def _fit_lambdarank(
    train: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict]:
    X_train, _, state = prepare_panel_matrix(train, feature_cols, "lambdarank", fit=True)
    rel_train = _compute_relevance(train)
    groups_train = train.groupby("target_date", sort=True).size().to_numpy()
    model = create_ranking_model(params)
    model.fit(X_train, rel_train, group=groups_train)
    return model, state


def _predict_lambdarank(
    model: Any,
    test: pd.DataFrame,
    feature_cols: list[str],
    state: dict,
) -> np.ndarray:
    X_test, _, _ = prepare_panel_matrix(test, feature_cols, "lambdarank", dummy_state=state, fit=False)
    return model.predict(X_test)


def _mean_rank_ic(preds: pd.DataFrame) -> float:
    ic = monthly_rank_ic(preds)
    if ic.empty or ic["rank_ic"].isna().all():
        return float("nan")
    return float(ic["rank_ic"].mean())


def _split_train_validation(
    train: pd.DataFrame,
    val_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(train["target_date"].unique())
    if len(dates) <= val_months + 6:
        mid = max(len(dates) // 2, 1)
        val_dates = set(dates[mid:])
        train_dates = set(dates[:mid])
    else:
        val_dates = set(dates[-val_months:])
        train_dates = set(dates[:-val_months])
    return (
        train[train["target_date"].isin(train_dates)],
        train[train["target_date"].isin(val_dates)],
    )


def select_rank_params_on_validation(
    train_inner: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    param_grid: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid search LambdaRank hyperparameters by validation Rank IC."""
    param_grid = param_grid or get_rank_param_grid()
    rows: list[dict] = []
    best_params: dict[str, Any] = {}
    best_ic = float("-inf")

    for params in param_grid:
        if train_inner.empty or val.empty:
            continue
        try:
            model, state = _fit_lambdarank(train_inner, feature_cols, params)
            scores = _predict_lambdarank(model, val, feature_cols, state)
            val_pred = val.copy()
            val_pred["y_pred"] = scores
            val_pred["model"] = "lambdarank"
            ic_mean = _mean_rank_ic(val_pred[["date", "y_pred", "y_actual", "model"]])
        except Exception:
            ic_mean = float("nan")

        row = {**params, "val_rank_ic_mean": ic_mean}
        rows.append(row)
        if not np.isnan(ic_mean) and ic_mean > best_ic:
            best_ic = ic_mean
            best_params = dict(params)

    log = pd.DataFrame(rows)
    if not best_params and not log.empty:
        best_row = log.sort_values("val_rank_ic_mean", ascending=False).iloc[0]
        best_params = {
            k: best_row[k]
            for k in ["n_estimators", "learning_rate", "num_leaves"]
            if k in best_row.index and pd.notna(best_row[k])
        }
    return best_params, log


def walk_forward_lambdarank_nested(
    df: pd.DataFrame | None = None,
    min_train_months: int | None = None,
    val_months: int | None = None,
    horizon: int = 1,
    train_mode: str = "expanding",
    rolling_window: int | None = None,
    panel: pd.DataFrame | None = None,
    save_selection_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nested walk-forward LambdaRank:
      Train_inner → Validation (select hyperparams) → Retrain on full train → OOS Test
    """
    cfg = get_ml_config()
    min_train_months = min_train_months or cfg.get("min_train_months", 36)
    val_months = val_months or cfg.get("validation", {}).get("val_months", 12)

    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    if panel is None:
        panel = build_long_panel(df, horizon=horizon)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_cols = get_panel_feature_columns(panel, include_industry_id=True)
    forecast_dates = sorted(panel["target_date"].unique())
    total = len(forecast_dates)
    records: list[dict] = []
    selection_rows: list[dict] = []

    print(f"  Nested walk-forward: {total} OOS months × 18 hyperparam trials each (~10–20 min)")
    import sys
    sys.stdout.flush()

    for idx, target_date in enumerate(forecast_dates):
        train_full = _select_train_panel(panel, target_date, train_mode, rolling_window, min_train_months)
        test = panel[panel["target_date"] == target_date]
        if train_full.empty or test.empty or len(test) < 3:
            continue

        train_inner, val = _split_train_validation(train_full, val_months)
        if train_inner.empty or val.empty:
            continue

        best_params, grid_log = select_rank_params_on_validation(train_inner, val, feature_cols)
        if not best_params:
            best_params = {}

        model, state = _fit_lambdarank(train_full, feature_cols, best_params)
        scores = _predict_lambdarank(model, test, feature_cols, state)

        val_ic = float("nan")
        if not grid_log.empty and "val_rank_ic_mean" in grid_log.columns:
            val_ic = float(grid_log["val_rank_ic_mean"].max())

        selection_rows.append(
            {
                "target_date": target_date,
                "Date": target_date.strftime("%Y-%m"),
                **best_params,
                "val_rank_ic_mean": val_ic,
                "n_train_months": train_full["target_date"].nunique(),
                "n_val_months": val["target_date"].nunique(),
            }
        )

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
                    "nested_validation": True,
                    **{f"hp_{k}": v for k, v in best_params.items()},
                }
            )

        if (idx + 1) % 3 == 0 or idx == total - 1:
            ic_txt = f"{val_ic:.3f}" if not np.isnan(val_ic) else "n/a"
            print(f"  [{idx + 1}/{total}] {target_date.strftime('%Y-%m')} done (val IC={ic_txt})")
            sys.stdout.flush()

    preds = pd.DataFrame(records)
    sel_log = pd.DataFrame(selection_rows)

    if save_selection_log and not sel_log.empty:
        out_dir = get_path("processed") / "ml" / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        sel_log.to_csv(out_dir / "hyperparam_selection_log.csv", index=False, encoding="utf-8-sig")

    return preds, sel_log


def summarize_validation_selection(sel_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate which hyperparameters were selected across OOS folds."""
    if sel_log.empty:
        return pd.DataFrame()
    hp_cols = [c for c in ["n_estimators", "learning_rate", "num_leaves"] if c in sel_log.columns]
    rows = []
    for col in hp_cols:
        counts = sel_log[col].value_counts()
        for val, cnt in counts.items():
            rows.append({"parameter": col, "value": val, "times_selected": int(cnt)})
    summary = pd.DataFrame(rows)
    if "val_rank_ic_mean" in sel_log.columns:
        summary.attrs["mean_val_rank_ic"] = float(sel_log["val_rank_ic_mean"].mean())
    return summary
