"""Feature filtering: Correlation → SHAP → Importance (3-step pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data.catalog import zscore_col
from src.ml.data import load_ml_dataset
from src.ml.features import prepare_panel_matrix
from src.ml.models import create_ranking_model
from src.ml.panel import build_long_panel, get_panel_feature_columns


def _correlation_filter(
    panel: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.85,
) -> tuple[list[str], pd.DataFrame]:
    X = panel[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr = X.corr().abs()
    drop: set[str] = set()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        if a in drop:
            continue
        for b in cols[i + 1 :]:
            if b in drop:
                continue
            if corr.loc[a, b] >= threshold:
                var_a = X[a].var()
                var_b = X[b].var()
                drop.add(a if var_a < var_b else b)
    kept = [c for c in feature_cols if c not in drop]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if corr.loc[a, b] >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "abs_corr": float(corr.loc[a, b])})
    return kept, pd.DataFrame(pairs)


def _importance_ranking(
    panel: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    X, cols, _ = prepare_panel_matrix(panel, feature_cols, "lambdarank", fit=True)
    rel = (
        panel.groupby("target_date")["y"]
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
    groups = panel.groupby("target_date", sort=True).size().to_numpy()
    model = create_ranking_model()
    model.fit(X, rel, group=groups)
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": cols, "gain_importance": np.zeros(len(cols))})
    return pd.DataFrame({"feature": cols, "gain_importance": model.feature_importances_}).sort_values(
        "gain_importance", ascending=False
    )


def _subsample_matrix(X: np.ndarray | pd.DataFrame, idx: np.ndarray) -> np.ndarray | pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return X[idx]


def _shap_ranking(
    panel: pd.DataFrame,
    feature_cols: list[str],
    max_samples: int = 800,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    X, cols, _ = prepare_panel_matrix(panel, feature_cols, "lambdarank", fit=True)
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_fit = _subsample_matrix(X, idx)
    else:
        X_fit = X

    model = create_ranking_model()
    rel = (
        panel.groupby("target_date")["y"]
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
    groups = panel.groupby("target_date", sort=True).size().to_numpy()
    model.fit(X, rel, group=groups)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_fit)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )


def run_feature_selection_pipeline(
    df: pd.DataFrame | None = None,
    corr_threshold: float | None = None,
    top_n: int | None = None,
) -> dict[str, pd.DataFrame | list[str]]:
    cfg = load_config().get("ml", {}).get("feature_selection", {})
    corr_threshold = corr_threshold if corr_threshold is not None else cfg.get("corr_threshold", 0.85)
    top_n = top_n or cfg.get("top_n", 35)

    df = df if df is not None else load_ml_dataset(use_training_sample=True)
    panel = build_long_panel(df)
    feature_cols = get_panel_feature_columns(panel, include_industry_id=True)

    after_corr, corr_pairs = _correlation_filter(panel, feature_cols, corr_threshold)
    gain = _importance_ranking(panel, after_corr)
    shap_df = _shap_ranking(panel, after_corr)

    merged = gain.merge(shap_df, on="feature", how="outer").fillna(0.0)
    merged["rank_gain"] = merged["gain_importance"].rank(ascending=False)
    merged["rank_shap"] = merged["mean_abs_shap"].rank(ascending=False)
    merged["combined_rank"] = merged["rank_gain"] + merged["rank_shap"]
    merged = merged.sort_values("combined_rank")

    n_gain = max(int(len(after_corr) * 0.6), 10)
    n_shap = max(int(len(after_corr) * 0.6), 10)
    top_gain = set(gain.head(n_gain)["feature"])
    top_shap = set(shap_df.head(n_shap)["feature"]) if not shap_df.empty else set()
    selected = list(dict.fromkeys([f for f in merged["feature"] if f in top_gain or f in top_shap]))[:top_n]
    if "industry_id" in feature_cols and "industry_id" not in selected:
        selected.append("industry_id")

    out_dir = get_path("processed") / "ml" / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_pairs.to_csv(out_dir / "correlation_pairs.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(out_dir / "gain_importance.csv", index=False, encoding="utf-8-sig")
    if not shap_df.empty:
        shap_df.to_csv(out_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "combined_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": selected}).to_csv(out_dir / "selected_features.csv", index=False, encoding="utf-8-sig")

    return {
        "initial": feature_cols,
        "after_correlation": after_corr,
        "selected": selected,
        "correlation_pairs": corr_pairs,
        "gain": gain,
        "shap": shap_df,
        "combined": merged,
    }
