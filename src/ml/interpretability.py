"""Model interpretability: SHAP, gain importance, category/yearly analysis, ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data.feature_catalog import CATEGORY_LABELS, classify_feature_category
from src.ml.features import prepare_panel_matrix
from src.ml.models import create_regression_model
from src.ml.panel import build_long_panel, get_panel_feature_columns


def _market_regime_label(df: pd.DataFrame) -> pd.Series:
    if "HS300_TREND_6M" in df.columns:
        trend = df.set_index("date")["HS300_TREND_6M"]
        return (trend > 0).map({True: "risk_on", False: "risk_off"})
    return pd.Series("all", index=df.set_index("date").index)


def _resolve_model_name(model_name: str | None = None) -> str:
    cfg = load_config().get("ml", {})
    if model_name:
        return model_name
    cmp_path = get_path("processed") / "ml" / "model_comparison.csv"
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path)
        if not cmp_df.empty and "sharpe" in cmp_df.columns:
            return str(cmp_df.sort_values("sharpe", ascending=False).iloc[0]["model"])
    return cfg.get("interpretability_model", "xgboost")


def _train_matrix(
    panel: pd.DataFrame,
    model_name: str,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.Series] | None:
    if feature_cols is None:
        feature_cols = get_panel_feature_columns(panel, include_industry_id=True)
    global_required = [c for c in feature_cols if not c.startswith("IND_DRV_") and c != "industry_id"]
    global_required = [c for c in global_required if c in panel.columns and panel[c].notna().any()]
    train = panel.dropna(subset=["y"] + global_required)
    if len(train) < 100:
        return None
    X, cols, _ = prepare_panel_matrix(train, feature_cols, model_name, fit=True)
    y = train["y"]
    return X, cols, train, y


def run_gain_importance(
    df: pd.DataFrame,
    model_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    output_dir = output_dir or get_path("processed") / "ml" / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _resolve_model_name(model_name)

    if model_name not in {"lightgbm", "xgboost"}:
        return pd.DataFrame()

    panel = build_long_panel(df)
    packed = _train_matrix(panel, model_name)
    if packed is None:
        return pd.DataFrame()
    X, cols, _, y = packed

    model = create_regression_model(model_name)
    model.fit(X, y.to_numpy(dtype=float))
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    out = pd.DataFrame({"feature": cols, "gain_importance": model.feature_importances_}).sort_values(
        "gain_importance", ascending=False
    )
    out.to_csv(output_dir / f"gain_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
    return out


def run_category_importance(
    df: pd.DataFrame,
    model_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Aggregate gain/SHAP importance by feature category."""
    output_dir = output_dir or get_path("processed") / "ml" / "interpretability"
    model_name = _resolve_model_name(model_name)

    gain = run_gain_importance(df, model_name, output_dir)
    shap_path = output_dir / f"shap_importance_{model_name}.csv"
    if shap_path.exists():
        imp = pd.read_csv(shap_path)
        val_col = "mean_abs_shap"
    elif not gain.empty:
        imp = gain
        val_col = "gain_importance"
    else:
        return pd.DataFrame()

    imp = imp.copy()
    imp["category"] = imp["feature"].map(classify_feature_category)
    imp["category_label"] = imp["category"].map(CATEGORY_LABELS)
    grouped = imp.groupby(["category", "category_label"], as_index=False)[val_col].sum()
    total = grouped[val_col].sum()
    grouped["share"] = grouped[val_col] / total if total else 0.0
    grouped = grouped.sort_values("share", ascending=False)
    grouped.to_csv(output_dir / f"category_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
    return grouped


def _friendly_feature_name(name: str) -> str:
    base = name.replace("_zscore", "")
    mapping = {
        "PMI": "PMI",
        "TSF": "社融",
        "NORTH_FLOW": "北向资金",
        "IND_MOM_3M": "行业3M动量",
        "IND_MOM_6M": "行业6M动量",
        "IND_REL_MOM_12M": "相对动量代理",
        "IND_RET_ACCEL": "收益加速代理",
        "IND_DRV_SEMI": "SOX半导体",
        "MARGIN_CHG": "融资变化",
        "HS300_TREND_6M": "HS300 6M趋势",
    }
    if base in mapping:
        return mapping[base]
    if base.startswith("IND_DRV_"):
        return base.replace("IND_DRV_", "") + "驱动"
    return base


def run_yearly_importance(
    df: pd.DataFrame,
    model_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Per-year top feature and dominant category (regime shift)."""
    output_dir = output_dir or get_path("processed") / "ml" / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _resolve_model_name(model_name)
    if model_name not in {"lightgbm", "xgboost"}:
        return pd.DataFrame()

    panel = build_long_panel(df)
    if panel.empty:
        return pd.DataFrame()

    feature_cols = get_panel_feature_columns(panel, include_industry_id=True)
    panel = panel.copy()
    panel["year"] = pd.to_datetime(panel["target_date"]).dt.year

    rows: list[dict] = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        if len(train) < 150:
            continue

        packed = _train_matrix(train, model_name, feature_cols)
        if packed is None:
            continue
        X, cols, _, y = packed
        model = create_regression_model(model_name)
        model.fit(X, y.to_numpy(dtype=float))
        if not hasattr(model, "feature_importances_"):
            continue

        imp = pd.DataFrame({"feature": cols, "gain_importance": model.feature_importances_})
        imp["category"] = imp["feature"].map(classify_feature_category)
        imp["category_label"] = imp["category"].map(CATEGORY_LABELS)

        top = imp.sort_values("gain_importance", ascending=False).iloc[0]
        cat = (
            imp.groupby(["category", "category_label"])["gain_importance"]
            .sum()
            .reset_index()
            .sort_values("gain_importance", ascending=False)
            .iloc[0]
        )
        rows.append(
            {
                "year": int(year),
                "top_feature": top["feature"],
                "top_feature_label": _friendly_feature_name(top["feature"]),
                "top_category": cat["category"],
                "top_category_label": cat["category_label"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / f"yearly_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
    return out


def run_shap_analysis(
    df: pd.DataFrame,
    model_name: str | None = None,
    max_samples: int = 800,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    output_dir = output_dir or get_path("processed") / "ml" / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _resolve_model_name(model_name)

    if model_name not in {"lightgbm", "xgboost"}:
        return {}

    panel = build_long_panel(df)
    packed = _train_matrix(panel, model_name)
    if packed is None:
        return {}
    X, cols, _, y = packed

    try:
        import shap
    except ImportError:
        return {}

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_fit, y_fit = X[idx], y.iloc[idx].to_numpy(dtype=float)
    else:
        X_fit, y_fit = X, y.to_numpy(dtype=float)

    model = create_regression_model(model_name)
    model.fit(X_fit, y_fit)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_fit)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    importance.to_csv(output_dir / f"shap_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
    return {"global": importance}


def run_regime_importance(
    df: pd.DataFrame,
    model_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    output_dir = output_dir or get_path("processed") / "ml" / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _resolve_model_name(model_name)

    panel = build_long_panel(df)
    if panel.empty:
        return pd.DataFrame()

    regimes = _market_regime_label(df)
    panel = panel.copy()
    panel["regime"] = panel["date"].map(regimes)
    feature_cols = get_panel_feature_columns(panel, include_industry_id=False)
    rows: list[dict] = []

    for regime, subset in panel.groupby("regime"):
        if len(subset) < 50:
            continue
        for col in feature_cols:
            clean = subset[[col, "y"]].dropna()
            if len(clean) < 30 or clean[col].std() == 0:
                continue
            corr = clean[col].corr(clean["y"])
            rows.append({"regime": regime, "feature": col, "corr_with_y": corr})

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / f"regime_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
    return out


def run_interpretability(df: pd.DataFrame | None = None) -> None:
    df = df if df is not None else __import__(
        "src.ml.data", fromlist=["load_ml_dataset"]
    ).load_ml_dataset(use_training_sample=True)
    model = _resolve_model_name()
    print(f"  Feature importance / SHAP for: {model}...")
    run_gain_importance(df, model_name=model)
    run_shap_analysis(df, model_name=model)
    run_category_importance(df, model_name=model)
    run_yearly_importance(df, model_name=model)
    run_regime_importance(df, model_name=model)

    from src.ml.ablation import run_ablation_study

    print("  Ablation study (cumulative feature blocks)...")
    ablation = run_ablation_study(df, model_name=model)
    if not ablation.empty:
        print(ablation[["stage_label", "sharpe", "rank_ic_mean"]].to_string(index=False))
