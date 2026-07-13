"""Curated feature sets to keep total dimensionality ~60-80 for small panel samples."""

from __future__ import annotations

from src.config import load_config
from src.data.catalog import zscore_col
from src.data.feature_catalog import DRIVER_COLS, SIGNAL_FEATURE_COLS
from src.fetch.industry_price_signals import INDUSTRY_PRICE_COLS

# --- Global features used in panel (~25 raw → ~25 z-scores) ---
SELECTED_MACRO = ["PMI", "CPI", "PPI", "M2", "TSF", "IND_PROD", "RETAIL", "BOND_10Y"]

SELECTED_LIQUIDITY = [
    "MARGIN_BAL",
    "NORTH_FLOW",
    "TURNOVER",
    "HS300_VOL",
    "MARGIN_CHG",
    "TURNOVER_ZSCORE",
]

SELECTED_MARKET = [
    "HS300_TREND_3M",
    "HS300_TREND_6M",
    "MARKET_BREADTH",
    "CYB_VS_HS300",
    "VOL_TREND",
    "HS300_MA20_GAP",
]

SELECTED_INDUSTRY_GLOBAL = [
    "IND_MOM_1M",
    "IND_MOM_3M",
    "IND_MOM_6M",
    "IND_REV_1M",
    "IND_DISP_1M",
]

SELECTED_GLOBAL_RAW = (
    SELECTED_MACRO + SELECTED_LIQUIDITY + SELECTED_MARKET + SELECTED_INDUSTRY_GLOBAL
)

# --- Per-industry panel features (~12-15) ---
SELECTED_INDUSTRY_PRICE = [
    "IND_PX_MOM_3M",
    "IND_PX_MOM_12M",
    "IND_REL_STRENGTH",
    "IND_VOL",
    "IND_POS_MONTH_RATIO",
]

SELECTED_INDUSTRY_SIGNALS = list(SIGNAL_FEATURE_COLS)

SELECTED_INDUSTRY_DRIVERS = [
    "IND_DRV_BANK_SPREAD",
    "IND_DRV_SEMI",
    "IND_DRV_HOUSING",
]

SELECTED_INDUSTRY_PANEL = (
    SELECTED_INDUSTRY_PRICE + SELECTED_INDUSTRY_SIGNALS + SELECTED_INDUSTRY_DRIVERS
)

# Momentum-only blocks for factor attribution
MOMENTUM_GLOBAL = list(SELECTED_INDUSTRY_GLOBAL)
MOMENTUM_PANEL = SELECTED_INDUSTRY_PRICE + [
    "IND_REL_MOM_12M",
    "IND_RET_ACCEL",
    "IND_MOM_CS_PCT",
]

MACRO_BLOCK = SELECTED_MACRO
LIQUIDITY_MARKET_BLOCK = SELECTED_LIQUIDITY + SELECTED_MARKET
DRIVER_BLOCK = SELECTED_INDUSTRY_DRIVERS


def get_selected_global_raw() -> list[str]:
    cfg = load_config()
    override = cfg.get("features", {}).get("selected_global")
    return list(override) if override else list(SELECTED_GLOBAL_RAW)


def get_selected_industry_panel() -> list[str]:
    cfg = load_config()
    override = cfg.get("features", {}).get("selected_industry_panel")
    return list(override) if override else list(SELECTED_INDUSTRY_PANEL)


def get_global_zscore_columns(df_columns: list[str] | None = None) -> list[str]:
    cols = []
    for name in get_selected_global_raw():
        z = zscore_col(name)
        if df_columns is None or z in df_columns:
            cols.append(z)
    return cols


def count_selected_features(include_industry_id: bool = True) -> dict[str, int]:
    n_global = len(get_selected_global_raw())
    n_industry = len(get_selected_industry_panel())
    n_id = 1 if include_industry_id else 0
    return {
        "global_zscore": n_global,
        "industry_panel": n_industry,
        "industry_id": n_id,
        "total_approx": n_global + n_industry + n_id,
    }
