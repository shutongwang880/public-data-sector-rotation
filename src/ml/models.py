"""Model factories for industry return prediction."""

from __future__ import annotations

import itertools
from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config


class ReturnModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def get_ml_config() -> dict:
    return load_config().get("ml", {})


def create_regression_model(name: str) -> Any:
    cfg = get_ml_config()
    rs = cfg.get("random_state", 42)

    if name == "ridge":
        alpha = cfg.get("ridge_alpha", 1.0)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=rs)),
            ]
        )
    if name == "lasso":
        alpha = cfg.get("lasso_alpha", 0.001)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=alpha, random_state=rs, max_iter=10000)),
            ]
        )
    if name == "lightgbm":
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": rs,
            "n_estimators": cfg.get("lightgbm_n_estimators", 200),
            "learning_rate": cfg.get("lightgbm_learning_rate", 0.05),
            "num_leaves": cfg.get("lightgbm_num_leaves", 15),
            "subsample": cfg.get("lightgbm_subsample", 0.8),
            "colsample_bytree": cfg.get("lightgbm_colsample_bytree", 0.8),
        }
        return lgb.LGBMRegressor(**params)
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=cfg.get("xgboost_n_estimators", 200),
            learning_rate=cfg.get("xgboost_learning_rate", 0.05),
            max_depth=cfg.get("xgboost_max_depth", 4),
            subsample=cfg.get("xgboost_subsample", 0.8),
            colsample_bytree=cfg.get("xgboost_colsample_bytree", 0.8),
            random_state=rs,
            verbosity=0,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=cfg.get("random_forest_n_estimators", 200),
                        max_depth=cfg.get("random_forest_max_depth", 6),
                        min_samples_leaf=cfg.get("random_forest_min_samples_leaf", 5),
                        random_state=rs,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown regression model: {name}")


def get_rank_param_grid() -> list[dict]:
    """Hyperparameter grid for nested validation (Rank IC on inner val set)."""
    cfg = get_ml_config()
    val_cfg = cfg.get("validation", {})
    if val_cfg.get("param_grid"):
        return list(val_cfg["param_grid"])

    n_est = val_cfg.get("n_estimators_grid", [100, 200, 300])
    lr = val_cfg.get("learning_rate_grid", [0.03, 0.05, 0.1])
    leaves = val_cfg.get("num_leaves_grid", [15, 31])
    grid = []
    for n, l, nl in itertools.product(n_est, lr, leaves):
        grid.append({"n_estimators": int(n), "learning_rate": float(l), "num_leaves": int(nl)})
    return grid


def create_ranking_model(params: dict | None = None) -> Any:
    import lightgbm as lgb

    cfg = get_ml_config()
    merged = {
        "n_estimators": cfg.get("rank_n_estimators", 200),
        "learning_rate": cfg.get("rank_learning_rate", 0.05),
        "num_leaves": cfg.get("rank_num_leaves", 15),
        "subsample": cfg.get("rank_subsample", 0.8),
        "colsample_bytree": cfg.get("rank_colsample_bytree", 0.8),
    }
    if params:
        merged.update({k: v for k, v in params.items() if k in merged or k in {"n_estimators", "learning_rate", "num_leaves"}})

    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(10)),
        n_estimators=int(merged["n_estimators"]),
        learning_rate=float(merged["learning_rate"]),
        num_leaves=int(merged["num_leaves"]),
        subsample=float(merged["subsample"]),
        colsample_bytree=float(merged["colsample_bytree"]),
        random_state=cfg.get("random_state", 42),
        verbosity=-1,
    )


def return_to_relevance(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Map continuous returns to integer relevance labels for ranking."""
    if len(y) < 2:
        return np.zeros(len(y), dtype=int)
    ranks = np.argsort(np.argsort(y))
    bins = np.floor(ranks / max(len(y) / n_bins, 1)).astype(int)
    return np.clip(bins, 0, n_bins - 1)
