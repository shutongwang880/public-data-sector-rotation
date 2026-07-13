# Public-Data Sector Rotation Decision Support System

**Research Report**

| Field | Detail |
|-------|--------|
| Report Date | 10 July 2026 |
| Sample Period | January 2020 – May 2026 |
| Out-of-Sample Window | February 2023 – May 2026 (40 months) |
| Primary Model | LambdaRank (LightGBM) |
| Portfolio Rule | Softmax-weighted Top-5 |
| Data Scope | Public sources only (AkShare); no Wind/CSMAR |

---

## Executive Summary

This report evaluates whether publicly available macro, liquidity, market-state, and sector-level information can support **cross-sectional ranking** of Shenwan Level-1 industries in the A-share market. The production pipeline applies nested walk-forward validation, trains a LambdaRank model on an expanding window, and constructs a monthly Top-5 portfolio using score-weighted (Softmax) allocation.

Over the 40-month out-of-sample period, the strategy delivers a net annualised return of **10.6%**, a Sharpe ratio of **0.47**, and a mean Rank IC of **0.071** (ICIR 0.30). The model outperforms the CSI 300 and equal-weight sector benchmarks on a risk-adjusted basis, but **does not exceed a simple 12-month momentum Top-5 rule** (Sharpe 0.53). The evidence is therefore best characterised as **preliminary**: the model demonstrates statistically meaningful ranking ability, yet price-based momentum remains a formidable baseline.

### Business Conclusion — May 2026 Application

Using features through **April 2026**, the model's applied output for **May 2026** is a ranked overweight set: **通信 (Telecommunications)**, **有色金属 (Non-ferrous Metals)**, **传媒 (Media)**, **国防军工 (Defence)**, and **机械设备 (Machinery)**. This is a **decision starting point**, not a claim of certain outperformance. Over the full OOS window, the Top-5 hit rate is **25.5%** — modest in absolute terms, but above a random null (~18.5%).

Ex post, the top-ranked sector (通信, +20.4%) validated the ranking direction; two of five overweight names delivered positive returns that month. A single month neither proves nor disproves the framework. The business value lies in converting recurring sector allocation into an **auditable, explainable monthly process** — with explicit benchmark comparison, regime governance, and momentum cross-check — rather than in pretending each monthly call will be correct.

---

## 1. Research Question

The central question is not *how much* each sector will return, but **which sectors are likely to outperform on a relative basis next month**. This framing aligns with practical asset-allocation workflows, where the decision is primarily about sector tilts rather than point forecasts of absolute returns.

Evaluation prioritises **Rank IC**, **ICIR**, and **Top-K hit rate** as primary metrics. Portfolio Sharpe ratio and drawdown are reported as secondary, implementation-oriented measures.

---

## 2. Data and Methodology

### 2.1 Data Sources

All inputs are drawn from public APIs (principally AkShare). The study deliberately excludes proprietary databases to ensure reproducibility. The comparable universe comprises **27 Shenwan Level-1 industries** after screening for data availability and structural breaks.

### 2.2 Feature Construction

Features are organised across five categories: macroeconomy, liquidity, market state, sector aggregates, and sector-specific price/signal proxies. Global variables are standardised (z-score); industry-level variables are aligned to a monthly panel.

**Publication lags** are enforced for macro releases. Features at month *t* are paired with sector returns at month *t+1*, preventing look-ahead bias. A minimum training window of 36 months is required before the first out-of-sample prediction.

### 2.3 Validation Design

Each out-of-sample month follows a nested procedure:

```
Inner Train → 12-Month Validation (hyperparameter selection) → Retrain on Full Train → Test
```

Hyperparameters (`n_estimators`, `learning_rate`, `num_leaves`) are selected by maximising validation Rank IC. The most frequently chosen configuration during OOS is `n_estimators = 100`, `learning_rate = 0.03`.

### 2.4 Portfolio Construction

The default production rule selects the Top-5 industries by model score and assigns weights via Softmax transformation, with a single-sector cap of 20%. Transaction costs are applied at 23 bps per one-way turnover.

---

## 3. Out-of-Sample Performance

### 3.1 Primary Strategy Metrics

| Metric | Value |
|--------|-------|
| Net Annual Return | 10.64% |
| Sharpe Ratio | 0.47 |
| Maximum Drawdown | −25.2% |
| Mean Rank IC | 0.071 |
| ICIR | 0.301 |
| Top-5 Hit Rate | 25.5% |
| Average Monthly Turnover | 75.6% |

![Cumulative NAV](../data/figures/ml/ml_nav_curves.png)

### 3.2 Benchmark Comparison

| Benchmark | Annual Return | Sharpe | Max Drawdown |
|-----------|--------------|--------|--------------|
| Strategy (LambdaRank + Softmax) | 10.64% | 0.47 | −25.2% |
| CSI 300 | 5.01% | 0.25 | −21.0% |
| Sector Equal Weight | 4.72% | 0.23 | −26.3% |
| Momentum Top-5 (12M) | 12.92% | **0.53** | −32.1% |
| Mean Reversion (12M) | −3.49% | −0.14 | −35.7% |
| Macro Rule (PMI) | −10.60% | −0.82 | −35.4% |
| Random Top-K (Null) | 4.09% | 0.19 | −27.1% |

The strategy exceeds passive and random benchmarks but trails the 12-month momentum rule. This gap should be disclosed in any business-facing application.

### 3.3 Sub-Period Analysis

| Period | Months | Sharpe |
|--------|--------|--------|
| 2023 | 11 | −1.61 |
| 2024 | 12 | 0.29 |
| 2025 | 12 | 1.82 |

Performance was materially negative in 2023, recovered in 2024, and strengthened in 2025. Regime sensitivity remains an open risk management concern.

### 3.4 Signal Persistence (IC Decay)

| Horizon | Mean Rank IC | ICIR |
|---------|-------------|------|
| 1 Month | 0.071 | 0.301 |
| 2 Months | −0.043 | −0.162 |
| 3 Months | −0.055 | −0.235 |

![IC Decay](../data/figures/ml/ml_ic_decay.png)

The ranking signal is concentrated at the one-month horizon, supporting a monthly rebalancing cadence. Predictive power deteriorates at longer horizons.

---

## 4. Current Forecast — May 2026

| Field | Value |
|-------|-------|
| Forecast Month | May 2026 |
| Feature Data As-Of | April 2026 |
| Model | LambdaRank |
| Horizon | 1 Month |

### 4.1 Recommended Overweight Sectors

| Rank | Sector (Shenwan L1) | Model Score | Signal | Suggested Weight |
|------|---------------------|-------------|--------|------------------|
| 1 | 通信 | 0.793 | Overweight | 21.5% |
| 2 | 有色金属 | 0.726 | Overweight | 21.5% |
| 3 | 传媒 | 0.567 | Overweight | 20.9% |
| 4 | 国防军工 | 0.478 | Overweight | 19.1% |
| 5 | 机械设备 | 0.357 | Overweight | 16.9% |

Weights reflect the Softmax scheme with a 20% single-sector cap. Actual May 2026 returns were available in the backtest file for ex-post reference: telecommunications (+20.4%) validated the top rank; non-ferrous metals, media, and defence posted negative realised returns, underscoring the probabilistic nature of cross-sectional forecasts.

### 4.2 Underweight Candidates

The bottom five sectors in the May 2026 ranking were 家用电器, 建筑装饰, 交通运输, 农林牧渔, and 汽车, classified as **Underweight** under the production rule.

---

## 5. Interpretation

### 5.1 Dominant Factors

Global SHAP analysis indicates that **12-month relative momentum**, **relative strength**, and **12-month price momentum** account for the largest share of model attributions. Macro variables (M2, PMI, margin balance changes) provide supplementary signal but do not dominate.

| Rank | Feature | Mean \|SHAP\| |
|------|---------|-------------|
| 1 | IND_REL_MOM_12M | 0.157 |
| 2 | industry_id | 0.112 |
| 3 | IND_REL_STRENGTH | 0.110 |
| 4 | IND_PX_MOM_12M | 0.105 |
| 5 | IND_MOM_CS_PCT | 0.085 |

![Feature Importance](../data/figures/ml/ml_feature_importance.png)

### 5.2 Practical Reading

The model should be positioned as a **structured decision-support layer** over public data, not as a standalone alpha source. Its principal value lies in (i) formalising sector ranking into a reproducible monthly process, (ii) integrating non-price public signals into a single score, and (iii) providing explainability through SHAP-based attribution. Where simplicity is paramount, the 12-month momentum benchmark remains competitive.

---

## 6. Limitations

1. **Short OOS sample.** Forty months of walk-forward testing limits statistical confidence; sub-period results are volatile.
2. **Public-data constraint.** Sparse sector fundamentals and event data reduce the information set relative to institutional databases.
3. **Momentum overlap.** Ranking performance partially reflects price-based factors already captured by simpler rules.
4. **Macro publication lag.** The training panel may trail raw feature files when latest-month macro releases are incomplete.
5. **Research use only.** Results do not account for implementation frictions beyond the stated transaction-cost assumption.

---

## 7. Reproducibility

```bash
python run_phase1.py --skip-fetch
python run_ml.py
python run_dashboard.py
```

Supporting documentation: [Business Analytics](BUSINESS_ANALYTICS.md) · [Data Pipeline](PIPELINE.md)

---

*Disclaimer: This document is prepared for academic research and business analytics demonstration. It does not constitute investment advice.*
