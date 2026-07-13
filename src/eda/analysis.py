"""Exploratory data analysis and visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import get_path, load_config
from src.data.catalog import ALL_FEATURE_COLS, FEATURE_CATEGORIES, MACRO_COLS
from src.data.industries import get_industry_columns

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_master_dataset() -> pd.DataFrame:
    path = get_path("processed") / "master_dataset.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    return_cols = get_industry_columns(df) + (["HS300"] if "HS300" in df.columns else [])
    cols = feature_cols + return_cols
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew()
    stats["kurtosis"] = df[cols].kurtosis()
    return stats


def plot_feature_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    groups = [(name, [c for c in cols if c in df.columns]) for name, cols in FEATURE_CATEGORIES.items()]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 12), sharex=True)
    for ax, (name, cols) in zip(axes, groups):
        for col in cols:
            ax.plot(df["date"], df[col], linewidth=1.0, label=col, alpha=0.85)
        ax.set_ylabel(name)
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Four-Category Feature Time Series", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_macro_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    macro_cols = [c for c in MACRO_COLS if c in df.columns]
    fig, axes = plt.subplots(len(macro_cols), 1, figsize=(12, 10), sharex=True)
    for ax, col in zip(axes, macro_cols):
        ax.plot(df["date"], df[col], linewidth=1.2)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Macro Indicators Time Series (2015-2025)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "macro_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_macro_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix (Four Categories)")
    fig.tight_layout()
    fig.savefig(output_dir / "macro_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_industry_return_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    exclude = set(ALL_FEATURE_COLS) | {"HS300", "Date", "date"}
    industry_cols = [c for c in df.columns if c not in exclude]
    melted = df.melt(id_vars=["date"], value_vars=industry_cols, var_name="industry", value_name="return")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(melted["return"].dropna(), bins=40, kde=True, ax=axes[0])
    axes[0].set_title("Industry Monthly Return Distribution")
    axes[0].set_xlabel("Monthly Return")

    vol = df[industry_cols].std().sort_values(ascending=False)
    vol.plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_title("Industry Return Volatility (Std Dev)")
    axes[1].set_ylabel("Std Dev")
    axes[1].tick_params(axis="x", rotation=75)

    fig.tight_layout()
    fig.savefig(output_dir / "industry_return_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns(df: pd.DataFrame, output_dir: Path) -> None:
    exclude = set(ALL_FEATURE_COLS) | {"Date", "date"}
    industry_cols = [c for c in df.columns if c not in exclude and c != "HS300"]

    cum = (1 + df[industry_cols + ["HS300"]]).cumprod()
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in industry_cols[:8]:
        ax.plot(df["date"], cum[col], label=col, alpha=0.8)
    ax.plot(df["date"], cum["HS300"], label="HS300", color="black", linewidth=2, linestyle="--")
    ax.set_title("Cumulative Returns (Sample Industries vs HS300)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(save: bool = True) -> dict[str, pd.DataFrame]:
    df = load_master_dataset()
    figures_dir = get_path("figures")
    processed_dir = get_path("processed")
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats = descriptive_statistics(df)
    if save:
        stats.to_csv(processed_dir / "descriptive_statistics.csv", encoding="utf-8-sig")
        plot_feature_timeseries(df, figures_dir)
        plot_macro_timeseries(df, figures_dir)
        plot_macro_correlation(df, figures_dir)
        plot_industry_return_distribution(df, figures_dir)
        plot_cumulative_returns(df, figures_dir)

    return {"master": df, "statistics": stats}
