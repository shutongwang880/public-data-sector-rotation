"""Orchestrate fetching of all four feature categories."""

from __future__ import annotations

import pandas as pd

from src.config import get_path
from src.data.catalog import ALL_FEATURE_COLS, FEATURE_CATEGORIES
from src.fetch.industry_factors import fetch_all_industry_factors
from src.fetch.liquidity_data import fetch_all_liquidity
from src.fetch.macro_data import fetch_all_macro
from src.fetch.market_state_data import fetch_all_market_state


def merge_feature_universe(
    macro: pd.DataFrame,
    liquidity: pd.DataFrame,
    market_state: pd.DataFrame,
    industry_factors: pd.DataFrame,
) -> pd.DataFrame:
    frames = [macro, liquidity, market_state, industry_factors]
    universe = frames[0]
    for frame in frames[1:]:
        universe = universe.merge(frame, on="date", how="outer")
    return universe.sort_values("date").reset_index(drop=True)


def refresh_global_features(save: bool = True) -> pd.DataFrame:
    """
    Recompute derived global features from cached raw CSV (no network).
    Useful with `run_phase1.py --skip-fetch`.
    """
    from src.fetch.liquidity_data import compute_margin_change, compute_turnover_zscore

    raw_dir = get_path("raw")
    macro = pd.read_csv(raw_dir / "macro_indicators.csv", parse_dates=["date"])
    industry_returns = pd.read_csv(raw_dir / "industry_monthly_returns.csv", parse_dates=["date"])

    liquidity_path = raw_dir / "feature_liquidity.csv"
    if liquidity_path.exists():
        liquidity = pd.read_csv(liquidity_path, parse_dates=["date"])
        for col in ["TURNOVER_ZSCORE", "MARGIN_CHG", "ETF_FLOW_MOM"]:
            if col in liquidity.columns:
                liquidity = liquidity.drop(columns=[col])
        liquidity = liquidity.merge(compute_turnover_zscore(liquidity), on="date", how="left")
        liquidity = liquidity.merge(compute_margin_change(liquidity), on="date", how="left")
        if save:
            liquidity.to_csv(liquidity_path, index=False, encoding="utf-8-sig")
    else:
        liquidity = fetch_all_liquidity(save=save)

    industry_factors = fetch_all_industry_factors(industry_returns, save=save)
    market_state = fetch_all_market_state(
        industry_returns=industry_returns,
        liquidity=liquidity,
        save=save,
    )
    universe = merge_feature_universe(macro, liquidity, market_state, industry_factors)
    if save:
        universe.to_csv(raw_dir / "feature_universe.csv", index=False, encoding="utf-8-sig")
    return universe


def fetch_all_features(
    industry_returns: pd.DataFrame | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    macro = fetch_all_macro(save=save)
    liquidity = fetch_all_liquidity(save=save)
    industry_factors = fetch_all_industry_factors(industry_returns, save=save)
    market_state = fetch_all_market_state(
        industry_returns=industry_returns,
        liquidity=liquidity,
        save=save,
    )
    universe = merge_feature_universe(macro, liquidity, market_state, industry_factors)

    if save:
        raw_dir = get_path("raw")
        universe.to_csv(raw_dir / "feature_universe.csv", index=False, encoding="utf-8-sig")

    return {
        "macro": macro,
        "liquidity": liquidity,
        "market_state": market_state,
        "industry_factors": industry_factors,
        "universe": universe,
        "categories": pd.DataFrame(
            [{"category": k, "features": ", ".join(v)} for k, v in FEATURE_CATEGORIES.items()]
        ),
        "all_feature_cols": ALL_FEATURE_COLS,
    }
