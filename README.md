# Public-Data Sector Rotation Decision Support System

[![CI](https://github.com/YOUR_USERNAME/public-data-sector-rotation/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/public-data-sector-rotation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A reproducible **decision-support system** for monthly Shenwan Level-1 sector rotation, built entirely on **public data** (AkShare). It delivers cross-sectional rankings, allocation weights, explainability, and an interactive dashboard — not black-box return forecasts.

```
Public Data → Feature Engineering → Nested Walk-Forward LambdaRank
    → Softmax Top-5 → Benchmark Evaluation → Dashboard + Word Report
```

---

## What It Does

| Layer | Output |
|-------|--------|
| **Data** | Macro, liquidity, market state, Shenwan L1 returns (public APIs only) |
| **Model** | LambdaRank cross-sectional ranking with nested walk-forward validation |
| **Portfolio** | Softmax-weighted Top-5 (20% single-sector cap) |
| **Delivery** | Streamlit dashboard · Word business report · Markdown research summary |

**Honest positioning:** The model shows meaningful ranking ability (Rank IC 0.071, Sharpe 0.47 OOS) but does **not** beat a simple 12-month momentum rule. It is designed as **structured decision support**, not a standalone alpha engine.

---

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch data + build features
python run_phase1.py

# 2. Train, backtest, generate Word report (~10–20 min)
python run_ml.py

# 3. Interactive dashboard
python run_dashboard.py
# → http://localhost:8501

# Full pipeline (one command)
python run_all.py --skip-fetch   # use cached raw data
```

---

## Project Structure

```
public-data-sector-rotation/
├── config/config.yaml          # dates, features, ML hyperparameters
├── run_phase1.py               # data ingestion + preprocessing
├── run_ml.py                   # modelling + backtest + report export
├── run_dashboard.py            # Streamlit UI
├── run_all.py                  # end-to-end orchestrator
├── generate_word.py            # Word business report (.docx)
├── src/
│   ├── fetch/                  # AkShare data collectors
│   ├── preprocess/             # cleaning, alignment, feature engineering
│   ├── ml/                     # LambdaRank, walk-forward, backtest, metrics
│   ├── dashboard/              # Streamlit app + analytics layer
│   └── report/                 # Word report generator
├── scripts/
│   └── validate_pipeline.py    # CI smoke test
├── docs/
│   ├── REPORT.md               # research summary (Markdown)
│   ├── BUSINESS_ANALYTICS.md   # stakeholder value + operating model
│   └── PIPELINE.md             # data flow + reproducibility
└── data/
    ├── raw/                    # fetched CSVs (gitignored)
    ├── processed/              # ML outputs (gitignored, generated locally)
    └── figures/                # charts (gitignored, generated locally)
```

---

## Commands

| Command | Output |
|---------|--------|
| `python run_phase1.py` | `data/processed/master_dataset_train.csv` |
| `python run_ml.py` | predictions, backtest, figures, `docs/REPORT.docx` |
| `python run_dashboard.py` | Streamlit UI at `:8501` |
| `python generate_word.py` | `docs/REPORT.docx` |
| `python run_all.py` | full pipeline |
| `python scripts/validate_pipeline.py` | CI smoke test |

---

## Documentation

- [Research Report](docs/REPORT.md) — OOS evidence and latest forecast
- [Business Analytics](docs/BUSINESS_ANALYTICS.md) — stakeholder value, operating model, decision playbook
- [Data Pipeline](docs/PIPELINE.md) — schema, scheduling, reproducibility

---

## Tech Stack

Python · LightGBM LambdaRank · AkShare · Streamlit · SHAP · Walk-Forward Validation · python-docx

---

## Disclaimer

Research prototype for academic and portfolio demonstration. Not investment advice. Out-of-sample evidence spans ~40 months; validate over longer horizons before production use.
