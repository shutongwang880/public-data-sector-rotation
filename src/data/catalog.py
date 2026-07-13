"""Feature universe definition across hierarchical information layers."""

from __future__ import annotations

from src.config import load_config

# Layer 1: Global Macro (core + optional extended from public sources)
MACRO_CORE_COLS = ["PMI", "CPI", "PPI", "M2", "TSF"]
MACRO_EXTENDED_COLS = [
    "IND_PROD",      # 工业增加值同比
    "FIXED_INV",     # 固定资产投资同比
    "RETAIL",        # 社会消费品零售同比
    "EXPORT_YOY",    # 出口同比
    "BOND_10Y",      # 十年国债收益率
    "FX_CNY",        # 人民币汇率 (USD/CNY)
    "UNEMPLOY",      # 城镇调查失业率
]
MACRO_COLS = MACRO_CORE_COLS + MACRO_EXTENDED_COLS

# Layer 2: Market Regime
MARKET_STATE_CORE_COLS = [
    "HS300_TREND_3M",
    "HS300_TREND_6M",
    "MARKET_BREADTH",
    "CYB_VS_HS300",
    "VOL_TREND",
]
MARKET_STATE_EXTENDED_COLS = [
    "HS300_MA20_GAP",
    "HS300_MA60_GAP",
    "HS300_VOL_20D",
    "HS300_VOL_60D",
    "HS300_VOL_CHG",
]
MARKET_STATE_COLS = MARKET_STATE_CORE_COLS + MARKET_STATE_EXTENDED_COLS

# Layer 3: Liquidity
LIQUIDITY_CORE_COLS = ["MARGIN_BAL", "NORTH_FLOW", "TURNOVER", "HS300_VOL"]
LIQUIDITY_EXTENDED_COLS = ["TURNOVER_ZSCORE", "MARGIN_CHG", "ETF_FLOW_MOM"]
LIQUIDITY_COLS = LIQUIDITY_CORE_COLS + LIQUIDITY_EXTENDED_COLS

# Layer 4: Industry price aggregates (market-level)
INDUSTRY_FACTOR_COLS = [
    "IND_MOM_1M",
    "IND_MOM_3M",
    "IND_MOM_6M",
    "IND_MOM_12M",
    "IND_REV_1M",
    "IND_DISP_1M",
]

# Hierarchical labels for reports
FEATURE_LAYERS = {
    "global_macro": MACRO_COLS,
    "market_regime": MARKET_STATE_COLS,
    "liquidity": LIQUIDITY_COLS,
    "industry_price_global": INDUSTRY_FACTOR_COLS,
}

FEATURE_CATEGORIES = {
    "macro": MACRO_COLS,
    "liquidity": LIQUIDITY_COLS,
    "market_state": MARKET_STATE_COLS,
    "industry_factors": INDUSTRY_FACTOR_COLS,
}

ALL_FEATURE_COLS = MACRO_COLS + LIQUIDITY_COLS + MARKET_STATE_COLS + INDUSTRY_FACTOR_COLS
CORE_FEATURE_COLS = (
    MACRO_CORE_COLS
    + LIQUIDITY_CORE_COLS
    + MARKET_STATE_CORE_COLS
    + INDUSTRY_FACTOR_COLS
)


def zscore_col(name: str) -> str:
    return f"{name}_zscore"


ZSCORE_COLS = [zscore_col(c) for c in ALL_FEATURE_COLS]


def get_publication_lags() -> dict[str, int]:
    cfg = load_config()
    base = dict(cfg.get("feature_lags", {}))
    release = cfg.get("publication_release_lags", {})
    for key, lag in release.items():
        base[key] = max(int(base.get(key, base.get("_default", 1))), int(lag))
    return base


def get_core_required_features() -> list[str]:
    cfg = load_config()
    core = cfg.get("features", {}).get("core_required")
    if core:
        return list(core)
    return list(CORE_FEATURE_COLS)


def get_required_features() -> list[str]:
    """Columns that must be non-null for master dataset rows."""
    cfg = load_config()
    required = cfg.get("features", {}).get("required_for_master")
    if required:
        return list(required)
    return get_core_required_features()


def get_optional_features() -> list[str]:
    cfg = load_config()
    optional = cfg.get("features", {}).get("optional_extended", [])
    extended = [c for c in ALL_FEATURE_COLS if c not in get_core_required_features()]
    return list(dict.fromkeys([*optional, *extended]))
