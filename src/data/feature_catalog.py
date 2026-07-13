"""Feature metadata: public-data industry signals, drivers, and ablation stages."""

from __future__ import annotations

# --- Public-data-based industry signals (price / commodity exposure; NOT Wind fundamentals) ---
SIGNAL_FEATURE_COLS = [
    "IND_REL_MOM_12M",
    "IND_RET_ACCEL",
    "IND_MOM_CS_PCT",
    "IND_COMMODITY",
]

# Backward-compatible alias
PROXY_FEATURE_COLS = SIGNAL_FEATURE_COLS

LEGACY_PROXY_ALIASES = {
    "IND_VAL_SPREAD": "IND_REL_MOM_12M",
    "IND_EARN_MOM": "IND_RET_ACCEL",
    "IND_PROSPERITY": "IND_MOM_CS_PCT",
}

SIGNAL_DESCRIPTIONS = {
    "IND_REL_MOM_12M": "Industry Relative Momentum Signal：12M行业收益 − HS300 12M",
    "IND_RET_ACCEL": "Industry Momentum Acceleration Signal：3M动量 − 12M动量/4",
    "IND_MOM_CS_PCT": "Industry Cross-sectional Momentum Signal：3M动量截面分位数",
    "IND_COMMODITY": "Industry Commodity Exposure Signal：映射期货1M收益；缺失时用PPI",
}

PROXY_DESCRIPTIONS = SIGNAL_DESCRIPTIONS

from src.fetch.industry_price_signals import INDUSTRY_PRICE_COLS  # noqa: E402

FUNDAMENTAL_COLS = ["IND_PE", "IND_PB", "IND_ROE", "IND_EPS_FY1"]

FUNDAMENTAL_DESCRIPTIONS = {
    "IND_PE": "行业市盈率（一致预期或成分股加权）",
    "IND_PB": "行业市净率",
    "IND_ROE": "行业净资产收益率",
    "IND_EPS_FY1": "一致预期EPS（FY1）",
}

DRIVER_COLS = [
    "IND_DRV_COAL_PRICE",
    "IND_DRV_INV_PROXY",
    "IND_DRV_SEMI",
    "IND_DRV_BANK_SPREAD",
    "IND_DRV_HOUSING",
]

DRIVER_DESCRIPTIONS = {
    "IND_DRV_COAL_PRICE": "Industry Driver：煤价/周期品（粗钢、动力煤等公开序列）",
    "IND_DRV_INV_PROXY": "Industry Driver：库存/投资周期（PPI、固投等公开代理）",
    "IND_DRV_SEMI": "Industry Driver：半导体景气（费城半导体 SOX）",
    "IND_DRV_BANK_SPREAD": "Industry Driver：银行利差（LPR − 存款基准利率）",
    "IND_DRV_HOUSING": "Industry Driver：地产周期（国房景气/商品房销售等）",
}

INDUSTRY_DRIVER_MAP: dict[str, list[str]] = {
    "IND_DRV_COAL_PRICE": ["煤炭", "钢铁", "石油石化", "基础化工"],
    "IND_DRV_INV_PROXY": ["煤炭", "钢铁", "有色金属", "基础化工", "建筑材料"],
    "IND_DRV_SEMI": ["电子", "计算机", "通信"],
    "IND_DRV_BANK_SPREAD": ["银行", "非银金融"],
    "IND_DRV_HOUSING": ["房地产", "建筑材料", "家用电器", "轻工制造"],
}

ALL_INDUSTRY_FEATURE_COLS = (
    INDUSTRY_PRICE_COLS + SIGNAL_FEATURE_COLS + FUNDAMENTAL_COLS + DRIVER_COLS
)

from src.data.catalog import (  # noqa: E402
    INDUSTRY_FACTOR_COLS,
    LIQUIDITY_COLS,
    MACRO_COLS,
    MARKET_STATE_COLS,
)

CATEGORY_LABELS = {
    "macro": "宏观",
    "liquidity": "流动性",
    "market_state": "市场状态",
    "industry_factors": "行业全局因子",
    "industry_price": "行业价格因子",
    "industry_signals": "公开数据行业信号",
    "industry_driver": "行业经济驱动",
    "industry_fundamental": "行业基本面",
    "industry_id": "行业固定效应",
    "other": "其他",
}


def classify_feature_category(feature_name: str) -> str:
    name = str(feature_name)
    if name == "industry_id" or name.startswith("ind_"):
        return "industry_id"
    base = name.replace("_zscore", "")
    if base in MACRO_COLS:
        return "macro"
    if base in LIQUIDITY_COLS:
        return "liquidity"
    if base in MARKET_STATE_COLS:
        return "market_state"
    if base in INDUSTRY_FACTOR_COLS:
        return "industry_factors"
    if base in INDUSTRY_PRICE_COLS:
        return "industry_price"
    if base in SIGNAL_FEATURE_COLS:
        return "industry_signals"
    if base in FUNDAMENTAL_COLS:
        return "industry_fundamental"
    if base in DRIVER_COLS or name.startswith("IND_DRV_"):
        return "industry_driver"
    return "other"


ABLATION_STAGES: list[tuple[str, str, list[str]]] = [
    ("macro", "Baseline（宏观）", ["PMI", "CPI", "PPI", "M2", "TSF"]),
    (
        "market",
        "+ 流动性/市场",
        ["PMI", "CPI", "PPI", "M2", "TSF", "MARGIN_BAL", "NORTH_FLOW", "TURNOVER", "HS300_VOL",
         "HS300_TREND_3M", "HS300_TREND_6M", "MARKET_BREADTH", "CYB_VS_HS300", "VOL_TREND"],
    ),
    (
        "industry_global",
        "+ 行业全局因子",
        ["PMI", "CPI", "PPI", "M2", "TSF", "MARGIN_BAL", "NORTH_FLOW", "TURNOVER", "HS300_VOL",
         "HS300_TREND_3M", "HS300_TREND_6M", "MARKET_BREADTH", "CYB_VS_HS300", "VOL_TREND",
         "IND_MOM_1M", "IND_MOM_3M", "IND_MOM_6M", "IND_REV_1M", "IND_DISP_1M"],
    ),
]


def normalize_proxy_columns(df):
    """Rename legacy proxy column names to current signal naming."""
    import pandas as pd

    out = df.copy()
    for old, new in LEGACY_PROXY_ALIASES.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    return out
