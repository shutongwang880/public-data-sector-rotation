"""Feature matrix preparation for panel models."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tree / boosting models keep feature names (avoids sklearn LGBM warning spam).
TREE_MODELS = {"lambdarank", "lightgbm", "xgboost", "random_forest", "rank"}


def _as_matrix(
    frame: pd.DataFrame,
    col_names: list[str],
    model_name: str,
) -> np.ndarray | pd.DataFrame:
    filled = frame.fillna(0.0)
    if model_name in TREE_MODELS:
        filled.columns = col_names
        return filled
    return filled.to_numpy(dtype=float)


def prepare_panel_matrix(
    panel: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    dummy_state: dict | None = None,
    fit: bool = True,
) -> tuple[np.ndarray | pd.DataFrame, list[str], dict]:
    """
    Prepare design matrix for pooled panel models.

    Linear models: one-hot encode industry_id to capture cross-industry heterogeneity.
    Tree models: keep industry_id as numeric/categorical feature.
    """
    state = dummy_state or {}
    X = panel[feature_cols].apply(pd.to_numeric, errors="coerce").copy()

    if "industry_id" not in X.columns:
        return _as_matrix(X, feature_cols, model_name), feature_cols, state

    if model_name in {"ridge", "lasso"}:
        id_col = X["industry_id"].astype(int)
        base_cols = [c for c in feature_cols if c != "industry_id"]
        base = X[base_cols]
        if fit:
            dummies = pd.get_dummies(id_col, prefix="ind", drop_first=True)
            state["dummy_columns"] = dummies.columns.tolist()
        else:
            dummies = pd.get_dummies(id_col, prefix="ind", drop_first=True)
            for col in state.get("dummy_columns", []):
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[state.get("dummy_columns", [])]
        out = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        col_names = out.columns.tolist()
        return _as_matrix(out, col_names, model_name), col_names, state

    return _as_matrix(X, feature_cols, model_name), feature_cols, state
