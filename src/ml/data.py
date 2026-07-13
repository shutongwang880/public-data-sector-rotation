"""Build supervised learning panels: X_t -> y_{t+1} per industry."""

from __future__ import annotations

import pandas as pd

from src.config import get_path, load_config
from src.data.catalog import ALL_FEATURE_COLS, ZSCORE_COLS, zscore_col
from src.data.feature_selection import get_global_zscore_columns, get_selected_industry_panel
from src.data.industry_mask import get_comparable_industry_columns


def get_ml_config() -> dict:
    return load_config().get("ml", {})


def get_feature_columns(df: pd.DataFrame, use_zscore: bool | None = None) -> list[str]:
    cfg = get_ml_config()
    use_zscore = cfg.get("use_zscore", True) if use_zscore is None else use_zscore

    if use_zscore:
        cols = get_global_zscore_columns(list(df.columns))
        if cols:
            return cols
    from src.data.feature_selection import get_selected_global_raw

    return [c for c in get_selected_global_raw() if c in df.columns]


def load_ml_dataset(use_training_sample: bool = True) -> pd.DataFrame:
    processed = get_path("processed")
    train_path = processed / "master_dataset_train.csv"
    full_path = processed / "master_dataset.csv"
    path = train_path if use_training_sample and train_path.exists() else full_path
    master = pd.read_csv(path, parse_dates=["date"])

    scaled_path = processed / "features_standardized.csv"
    if scaled_path.exists():
        scaled = pd.read_csv(scaled_path, parse_dates=["date"])
        z_cols = [c for c in scaled.columns if c.endswith("_zscore")]
        master = master.merge(scaled[["date", *z_cols]], on="date", how="left")

    return master.sort_values("date").reset_index(drop=True)


def get_ml_industries(df: pd.DataFrame) -> list[str]:
    cfg = get_ml_config()
    min_obs = cfg.get("min_industry_obs", 48)
    industries = get_comparable_industry_columns(df)
    return [c for c in industries if df[c].notna().sum() >= min_obs]


def build_industry_panels(
    df: pd.DataFrame,
    industries: list[str] | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Per-industry panel with aligned (X_t, y_{t+1}).

    Row at date t uses features known at end of month t to predict return at t+1.
    """
    industries = industries or get_ml_industries(df)
    feature_cols = feature_cols or get_feature_columns(df)
    panels: dict[str, pd.DataFrame] = {}

    for industry in industries:
        rows: list[dict] = []
        for i in range(len(df) - 1):
            x_row = df.iloc[i]
            y_next = df.iloc[i + 1][industry]
            if pd.isna(y_next):
                continue
            feats = x_row[feature_cols]
            if feats.isna().any():
                continue
            rows.append(
                {
                    "Date": x_row.get("Date", x_row["date"].strftime("%Y-%m")),
                    "date": x_row["date"],
                    "target_date": df.iloc[i + 1]["date"],
                    "industry": industry,
                    "y": float(y_next),
                    **{c: float(feats[c]) for c in feature_cols},
                }
            )
        if rows:
            panels[industry] = pd.DataFrame(rows)
    return panels


def build_ranking_panel(
    df: pd.DataFrame,
    industries: list[str] | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pooled panel for LambdaRank: one query (group) per forecast month."""
    industries = industries or get_ml_industries(df)
    feature_cols = feature_cols or get_feature_columns(df)
    panels = build_industry_panels(df, industries, feature_cols)

    frames = []
    for idx, industry in enumerate(industries):
        if industry not in panels:
            continue
        part = panels[industry].copy()
        part["industry_id"] = idx
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["target_date", "industry"])
