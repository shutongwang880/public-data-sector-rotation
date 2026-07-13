"""ML strategy visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_path

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_nav_curves(backtests: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, bt in backtests.items():
        if bt.empty:
            continue
        ax.plot(bt["date"], bt["nav"], label=name, linewidth=1.5)
    sample = next(iter(backtests.values()))
    if not sample.empty:
        ax.plot(sample["date"], sample["hs300_nav"], label="沪深300", linestyle="--", color="gray")
        if "csi500_nav" in sample.columns:
            ax.plot(sample["date"], sample["csi500_nav"], label="中证500", linestyle="--", color="darkorange")
        if "ew_nav" in sample.columns:
            ax.plot(sample["date"], sample["ew_nav"], label="行业等权", linestyle=":", color="green")
    ax.set_title("Cumulative Return (Walk-Forward Top-K)")
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ml_nav_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_series(ic_df: pd.DataFrame, output_dir: Path) -> None:
    if ic_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ic_col = "rank_ic" if "rank_ic" in ic_df.columns else "ic"
    for model, group in ic_df.groupby("model"):
        ax.plot(group["date"], group[ic_col], label=model, alpha=0.8, linewidth=1.0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Monthly Rank IC")
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ml_ic_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_turnover(backtest: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    if backtest.empty or "turnover" not in backtest.columns:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(backtest["date"], backtest["turnover"], width=20, color="steelblue", alpha=0.75)
    ax.set_title(f"Monthly Turnover ({model_name})")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "ml_turnover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model_name: str, output_dir: Path, top_n: int = 10) -> None:
    interp = get_path("processed") / "ml" / "interpretability"
    shap_path = interp / f"shap_importance_{model_name}.csv"
    gain_path = interp / f"gain_importance_{model_name}.csv"
    path = shap_path if shap_path.exists() else gain_path
    if not path.exists():
        return
    df = pd.read_csv(path).head(top_n)
    val_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else "gain_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["feature"][::-1], df[val_col][::-1], color="darkorange", alpha=0.85)
    ax.set_title(f"Top-{top_n} Feature Importance ({model_name})")
    ax.set_xlabel(val_col)
    fig.tight_layout()
    fig.savefig(output_dir / "ml_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_importance(model_name: str, output_dir: Path) -> None:
    path = get_path("processed") / "ml" / "interpretability" / f"category_importance_{model_name}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "share" not in df.columns:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = df["category_label"] if "category_label" in df.columns else df["category"]
    ax.pie(df["share"], labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Category Importance Share")
    fig.tight_layout()
    fig.savefig(output_dir / "ml_category_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_turnover_sweep(portfolio_df: pd.DataFrame, output_dir: Path, label: str = "") -> None:
    if portfolio_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for weighting, group in portfolio_df.groupby("weighting"):
        ax.plot(group["avg_turnover"], group["sharpe"], marker="o", label=weighting)
    ax.set_xlabel("Average Turnover")
    ax.set_ylabel("Sharpe")
    ax.set_title(f"Sharpe vs Turnover ({label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    fig.savefig(output_dir / f"ml_turnover_sweep{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stability_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, analysis, title in zip(
        axes,
        ["top_k", "horizon", "train_window"],
        ["Top-K Sensitivity", "Forecast Horizon", "Training Window"],
    ):
        sub = summary[summary["analysis"] == analysis]
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            continue
        x = sub["variant"].astype(str)
        ax.bar(x, sub["sharpe"], color="steelblue", alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Sharpe")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "ml_stability_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_decay(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty or "horizon_months" not in summary.columns:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = summary.sort_values("horizon_months")
    ax.plot(sub["horizon_months"], sub["rank_ic_mean"], marker="o", linewidth=2, color="darkred")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Forward Horizon (months)")
    ax.set_ylabel("Mean Rank IC")
    ax.set_title("IC Decay (same 1M model vs multi-horizon actual)")
    ax.set_xticks(sub["horizon_months"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ml_ic_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_ml_figures(
    backtests: dict[str, pd.DataFrame],
    ic_df: pd.DataFrame,
    output_dir: Path,
    best_model: str | None = None,
) -> None:
    plot_nav_curves(backtests, output_dir)
    plot_ic_series(ic_df, output_dir)
    if best_model and best_model in backtests:
        plot_turnover(backtests[best_model], best_model, output_dir)
    if best_model:
        plot_feature_importance(best_model, output_dir)
        plot_category_importance(best_model, output_dir)
