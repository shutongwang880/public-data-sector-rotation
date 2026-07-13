"""Portfolio construction with weight caps, vol-adjusted, risk parity, turnover penalty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def get_risk_config() -> dict:
    ml = load_config().get("ml", {})
    return ml.get("risk_control", {})


def score_with_turnover_penalty(
    group: pd.DataFrame,
    prev_weights: dict[str, float],
    lambda_turnover: float,
) -> pd.DataFrame:
    if lambda_turnover <= 0 or not prev_weights:
        out = group.copy()
        out["score"] = out["y_pred"]
        return out

    rows: list[dict] = []
    for _, row in group.iterrows():
        ind = row["industry"]
        old_w = prev_weights.get(ind, 0.0)
        turnover_est = abs(1.0 - old_w)
        score = float(row["y_pred"]) - lambda_turnover * turnover_est
        rows.append({**row.to_dict(), "score": score})
    return pd.DataFrame(rows)


def select_top_k_scored(
    group: pd.DataFrame,
    top_k: int,
    score_col: str = "score",
) -> list[str]:
    valid = group.dropna(subset=[score_col])
    if valid.empty:
        return []
    return valid.nlargest(min(top_k, len(valid)), score_col)["industry"].tolist()


def _cap_and_normalize(raw: dict[str, float], max_weight: float) -> dict[str, float]:
    capped = {k: min(v, max_weight) for k, v in raw.items()}
    s = sum(capped.values())
    return {k: v / s for k, v in capped.items()} if s > 0 else raw


def _inverse_vol_weights(
    industries: list[str],
    vol_map: dict[str, float],
    max_weight: float,
    power: float = 1.0,
) -> dict[str, float]:
    inv = []
    for ind in industries:
        vol = vol_map.get(ind)
        if vol is None or pd.isna(vol) or vol <= 0:
            inv.append(1.0)
        else:
            inv.append(1.0 / (vol**power))
    total = sum(inv)
    if total <= 0:
        return {ind: 1.0 / len(industries) for ind in industries}
    raw = {ind: w / total for ind, w in zip(industries, inv)}
    return _cap_and_normalize(raw, max_weight)


def _equal_weights(industries: list[str], max_weight: float) -> dict[str, float]:
    if not industries:
        return {}
    raw = {ind: 1.0 / len(industries) for ind in industries}
    return _cap_and_normalize(raw, max_weight)


def _softmax_weights(
    industries: list[str],
    score_map: dict[str, float],
    max_weight: float,
    temperature: float = 1.0,
) -> dict[str, float]:
    if not industries:
        return {}
    temperature = max(float(temperature), 1e-6)
    scores = np.array([float(score_map.get(ind, 0.0)) for ind in industries], dtype=float)
    scores = scores - scores.max()
    exp_s = np.exp(scores / temperature)
    total = exp_s.sum()
    if total <= 0:
        return _equal_weights(industries, max_weight)
    raw = {ind: float(w) for ind, w in zip(industries, exp_s / total)}
    return _cap_and_normalize(raw, max_weight)


def construct_portfolio_weights(
    industries: list[str],
    vol_map: dict[str, float] | None = None,
    method: str | None = None,
    max_weight: float | None = None,
    score_map: dict[str, float] | None = None,
    softmax_temperature: float | None = None,
) -> dict[str, float]:
    cfg = get_risk_config()
    method = method or cfg.get("default_weighting", "softmax")
    max_weight = max_weight if max_weight is not None else cfg.get("max_industry_weight", 0.20)
    temperature = softmax_temperature if softmax_temperature is not None else cfg.get("softmax_temperature", 1.0)

    if not industries:
        return {}

    if method == "softmax" and score_map:
        return _softmax_weights(industries, score_map, max_weight, temperature)

    if method == "equal":
        return _equal_weights(industries, max_weight)

    if method == "vol_adjusted" and vol_map:
        return _inverse_vol_weights(industries, vol_map, max_weight, power=0.5)

    if method == "risk_parity" and vol_map:
        return _inverse_vol_weights(industries, vol_map, max_weight, power=1.0)

    return _equal_weights(industries, max_weight)


def portfolio_return(weights: dict[str, float], returns_row: pd.Series) -> float:
    total = 0.0
    for ind, w in weights.items():
        r = returns_row.get(ind)
        if pd.notna(r):
            total += w * float(r)
    return total


def turnover(old_w: dict[str, float], new_w: dict[str, float]) -> float:
    keys = set(old_w) | set(new_w)
    return 0.5 * sum(abs(new_w.get(k, 0.0) - old_w.get(k, 0.0)) for k in keys)
