"""
Decision Support Dashboard — A-Share Sector Rotation
Designed for Business Analytics / FinTech portfolio presentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import analytics as az
from src.dashboard import data_loader as dl
from src.dashboard.theme import inject_theme, kpi_row, page_header

st.set_page_config(
    page_title="Sector Rotation · Decision Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d8dee9",
    "axes.labelcolor": "#5c6b7f",
    "text.color": "#1a2332",
    "xtick.color": "#5c6b7f",
    "ytick.color": "#5c6b7f",
    "grid.color": "#e8ecf0",
    "font.sans-serif": ["IBM Plex Sans", "Arial Unicode MS", "SimHei"],
    "axes.unicode_minus": False,
})

SIGNAL_CLASS = {
    "Overweight": "rank-overweight",
    "Neutral": "rank-neutral",
    "Underweight": "rank-underweight",
}


def _pct(v, d=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v) * 100:.{d}f}%"


def _num(v, d=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):.{d}f}"


def _fmt_val(v, unit=""):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if abs(float(v)) > 1e8:
        return f"{float(v)/1e8:.1f} bn"
    if unit == "%":
        return f"{float(v):.1f}%"
    return f"{float(v):.2f}"


# ── Page 1: Executive Overview ────────────────────────────────────────────────

def page_executive():
    page_header("Executive Overview", "Investment Recommendation · Decision Summary")

    fc = dl.load_live_forecast()
    market = az.load_market_environment()
    shap = az.load_global_shap()

    if not fc:
        st.error("No forecast available. Run `python run_phase1.py` then `python run_ml.py`.")
        return

    regime = market.get("regime", {}) if market else {}

    kpi_row([
        ("Forecast Month", fc.get("forecast_month", "—")),
        ("Data As-Of", fc.get("data_as_of", "—")),
        ("Model", str(fc.get("model", "LambdaRank")).upper()),
        ("Horizon", "1 Month"),
        ("Today", fc.get("today", "—")),
    ])

    st.markdown(
        f'<div class="hero-card"><h3>MARKET REGIME</h3><p>{regime.get("headline_regime", "—")}</p></div>',
        unsafe_allow_html=True,
    )

    pills = "".join(
        f'<span class="regime-pill">{v}</span>'
        for v in [regime.get("liquidity"), regime.get("risk"), regime.get("growth"), regime.get("momentum")]
        if v
    )
    if pills:
        st.markdown(pills, unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Sector Ranking · Investment Recommendation")
        df = pd.DataFrame(fc["rankings"])
        show = df[["rank", "industry", "score", "signal"]].copy()
        show.columns = ["Rank", "Sector", "Score", "Signal"]
        show["Score"] = show["Score"].map(lambda x: _num(x, 3))

        def _highlight(row):
            cls = SIGNAL_CLASS.get(row["Signal"], "rank-neutral")
            return [""] * 3 + [cls]

        st.dataframe(show, use_container_width=True, hide_index=True, height=480)

    with right:
        st.markdown("#### Portfolio Allocation")
        schemes = az.load_portfolio_schemes(fc)
        scheme_names = [s["label"] for s in schemes]
        selected = st.radio("Weighting Scheme", scheme_names, index=0, label_visibility="collapsed")

        scheme = next(s for s in schemes if s["label"] == selected)
        wdf = pd.DataFrame(scheme["weights"])
        if not wdf.empty:
            wdf["weight"] = wdf["weight"].map(lambda x: _pct(x))
            wdf.columns = ["Sector", "Weight"]
            st.dataframe(wdf, use_container_width=True, hide_index=True)

        if scheme.get("primary"):
            st.caption("★ Primary recommendation")

    summary = az.generate_decision_summary(fc, market, shap)
    st.markdown(f'<div class="decision-summary">{summary}</div>', unsafe_allow_html=True)


# ── Page 2: Market Environment ────────────────────────────────────────────────

def page_market():
    page_header("Market Environment", "Why the model predicts what it predicts")

    env = az.load_market_environment()
    if not env:
        st.warning("Market feature data not found.")
        return

    st.markdown(f"**Data as of:** {env['as_of']}")
    regime = env.get("regime", {})
    st.markdown(
        f"**Current Regime:** {regime.get('headline_regime', '—')}",
    )

    for module, indicators in env.get("modules", {}).items():
        st.markdown(f"#### {module}")
        cols = st.columns(len(indicators))
        for col, ind in zip(cols, indicators):
            with col:
                z = ind.get("zscore")
                delta = f"z={_num(z, 2)}" if z is not None and not pd.isna(z) else None
                st.metric(
                    ind["label"],
                    _fmt_val(ind.get("value"), ind.get("unit", "")),
                    delta=delta,
                )

    st.markdown("#### Indicator Trends (24M)")
    history = env.get("history", pd.DataFrame())
    if not history.empty:
        tabs = st.tabs(["PMI", "Liquidity", "Market Sentiment"])
        with tabs[0]:
            if "PMI" in history.columns:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.plot(history["date"], history["PMI"], color="#4d9fff", linewidth=2)
                ax.set_title("PMI")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
        with tabs[1]:
            fig, ax = plt.subplots(figsize=(9, 3))
            if "MARGIN_BAL_zscore" in history.columns:
                ax.plot(history["date"], history["MARGIN_BAL_zscore"], label="Margin Balance", color="#c9a227")
            if "NORTH_FLOW_zscore" in history.columns:
                ax.plot(history["date"], history["NORTH_FLOW_zscore"], label="Northbound Flow", color="#3dd68c")
            ax.axhline(0, color="#8b9cb8", linestyle="--", linewidth=0.8)
            ax.legend()
            ax.set_title("Liquidity (Standardized)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        with tabs[2]:
            fig, ax = plt.subplots(figsize=(9, 3))
            if "TURNOVER_zscore" in history.columns:
                ax.plot(history["date"], history["TURNOVER_zscore"], label="Turnover", color="#4d9fff")
            if "HS300_VOL_zscore" in history.columns:
                ax.plot(history["date"], history["HS300_VOL_zscore"], label="HS300 Vol", color="#f07178")
            ax.axhline(0, color="#8b9cb8", linestyle="--", linewidth=0.8)
            ax.legend()
            ax.set_title("Market Sentiment (Standardized)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ── Page 3: Sector Intelligence ─────────────────────────────────────────────

def page_sector():
    page_header("Sector Intelligence", "Drill-down analysis by industry")

    fc = dl.load_live_forecast()
    if not fc:
        st.warning("No forecast data.")
        return

    industries = [r["industry"] for r in fc["rankings"]]
    default = industries[0] if industries else None
    selected = st.selectbox("Select Sector", industries, index=0)

    intel = az.load_sector_intelligence(selected)
    if not intel:
        st.warning("No data for this sector.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Rank", f"#{intel.get('rank', '—')}")
    c2.metric("Prediction Score", _num(intel.get("latest_score"), 3))
    c3.metric("Last Month Return", _pct(intel.get("latest_actual")))

    st.markdown(f'<div class="decision-summary">{intel.get("explanation", "")}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Score & Return History")
        hist = pd.DataFrame(intel.get("history", []))
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(hist["date"], hist["y_pred"], label="Prediction Score", color="#4d9fff", marker="o", markersize=3)
            if hist["y_actual"].notna().any():
                ax.bar(hist["date"], hist["y_actual"], alpha=0.35, label="Actual Return", color="#3dd68c")
            ax.axhline(0, color="#8b9cb8", linewidth=0.6)
            ax.legend(fontsize=8)
            ax.set_title(f"{selected} · 12M History")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.markdown("#### Key Drivers (Local SHAP)")
        drivers = pd.DataFrame(intel.get("shap_drivers", []))
        if not drivers.empty:
            colors = ["#3dd68c" if d == "positive" else "#f07178" for d in drivers["direction"]]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(drivers["feature"], drivers["shap"].abs(), color=colors, alpha=0.85)
            ax.set_title("Feature Contribution")
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("No SHAP case study for this sector yet.")


# ── Page 4: Explainable AI ───────────────────────────────────────────────────

def page_xai():
    page_header("Explainable AI", "Model interpretability · Feature attribution")

    shap = az.load_global_shap()
    fc = dl.load_live_forecast()

    if shap.get("features"):
        top3 = shap["features"][:3]
        biz_text = "、".join(f["label"] for f in top3)
        st.markdown(
            f'<div class="decision-summary">'
            f"Sector rankings for the current month are primarily shaped by "
            f"<b>{biz_text}</b>, with <b>{top3[0]['label']}</b> contributing the most."
            f"</div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Global Feature Importance")
        feats = pd.DataFrame(shap.get("features", []))
        if not feats.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(feats["label"], feats["importance"], color="#4d9fff", alpha=0.85)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP|")
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.markdown("#### Factor Category Attribution")
        cats = pd.DataFrame(shap.get("categories", []))
        if not cats.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.pie(cats["share"], labels=cats["category"], autopct="%1.0f%%", colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(cats))))
            ax.set_title("Category Share")
            st.pyplot(fig)
            plt.close(fig)

    if fc:
        st.markdown("#### Local SHAP · Top Recommended Sector")
        top_ind = fc["top_holdings"][0]["industry"] if fc.get("top_holdings") else None
        if top_ind:
            local = az.load_local_shap(top_ind)
            if local:
                st.caption(f"Sector: **{top_ind}**")
                ldf = pd.DataFrame(local)
                st.dataframe(ldf, use_container_width=True, hide_index=True)


# ── Page 5: Portfolio Performance ───────────────────────────────────────────

def page_performance():
    page_header("Portfolio Performance", "Strategy evaluation as an investable proposition")

    metrics = dl.load_primary_metrics()
    bench = dl.load_benchmarks()
    bt = dl.load_backtest()
    experiments = az.load_portfolio_experiments()
    topk = az.load_topk_stability()

    if metrics:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Annual Return", _pct(metrics.get("annual_return")))
        c2.metric("Sharpe Ratio", _num(metrics.get("sharpe")))
        c3.metric("Max Drawdown", _pct(metrics.get("max_drawdown")))
        c4.metric("Rank IC", _num(metrics.get("rank_ic_mean"), 3))
        c5.metric("Top-K Hit Rate", _pct(metrics.get("top_k_hit_rate")))
        c6.metric("Avg Turnover", _pct(metrics.get("avg_turnover")))

    if not bt.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bt["date"], bt["nav"], label="Strategy", color="#c9a227", linewidth=2)
        ax.plot(bt["date"], bt["hs300_nav"], label="CSI 300", color="#8b9cb8", linestyle="--")
        ax.plot(bt["date"], bt["ew_nav"], label="Sector EW", color="#4d9fff", linestyle=":")
        ax.set_title("Cumulative NAV · Out-of-Sample")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Benchmark Comparison")
        if not bench.empty:
            show = bench[["benchmark", "annual_return", "sharpe", "max_drawdown"]].copy()
            show["annual_return"] = show["annual_return"].map(_pct)
            show["max_drawdown"] = show["max_drawdown"].map(_pct)
            show["sharpe"] = show["sharpe"].map(_num)
            show.columns = ["Benchmark", "Annual Return", "Sharpe", "Max DD"]
            st.dataframe(show, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Weighting Scheme Comparison")
        if not experiments.empty:
            exp = experiments.copy()
            exp["annual_return"] = exp["annual_return"].map(_pct)
            exp["max_drawdown"] = exp["max_drawdown"].map(_pct)
            exp["sharpe"] = exp["sharpe"].map(_num)
            exp["avg_turnover"] = exp["avg_turnover"].map(_pct)
            st.dataframe(exp, use_container_width=True, hide_index=True)

    if not topk.empty:
        st.markdown("#### Top-K Sensitivity")
        tk = topk.copy()
        tk["annual_return"] = tk["annual_return"].map(_pct)
        tk["sharpe"] = tk["sharpe"].map(_num)
        st.dataframe(tk[["variant", "annual_return", "sharpe", "top_k_hit_rate"]], hide_index=True)


# ── Page 6: Model Evaluation ─────────────────────────────────────────────────

def page_model_eval():
    page_header("Model Evaluation", "Model selection process · Why LambdaRank")

    eval_df = az.load_model_evaluation()
    if eval_df.empty:
        st.warning("Model comparison data not found.")
        return

    st.markdown(
        "Nested Walk-Forward validation selects hyperparameters on a 12-month "
        "validation window. **LambdaRank** is the production model for cross-sectional ranking."
    )

    show = eval_df.copy()
    if "Annual Return" in show.columns:
        show["Annual Return"] = show["Annual Return"].map(_pct)
    if "Max Drawdown" in show.columns:
        show["Max Drawdown"] = show["Max Drawdown"].map(_pct)
    if "Rank IC" in show.columns:
        show["Rank IC"] = show["Rank IC"].map(lambda x: _num(x, 3))
    if "Top-K Hit Rate" in show.columns:
        show["Top-K Hit Rate"] = show["Top-K Hit Rate"].map(_pct)
    if "Sharpe" in show.columns:
        show["Sharpe"] = show["Sharpe"].map(_num)

    st.dataframe(show, use_container_width=True, hide_index=True)

    if "Sharpe" in eval_df.columns and "Model" in eval_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#c9a227" if m == "lambdarank" else "#4d9fff" for m in eval_df["Model"]]
        ax.bar(eval_df["Model"], eval_df["Sharpe"], color=colors, alpha=0.85)
        ax.set_title("Sharpe Ratio by Model")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    ic = dl.load_ic_decay()
    if not ic.empty:
        st.markdown("#### Signal Decay (IC Persistence)")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(ic["horizon_months"].astype(str) + "M", ic["rank_ic_mean"], color="#e6b450", alpha=0.85)
        ax.axhline(0, color="#8b9cb8", linestyle="--")
        ax.set_ylabel("Mean Rank IC")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Signal is most effective at 1-month horizon — supports monthly rebalancing.")


# ── Page 7: Data Pipeline ─────────────────────────────────────────────────────

def page_pipeline():
    page_header("Data Pipeline", "End-to-end analytics lifecycle")

    steps = az.get_pipeline_steps()

    st.markdown(
        """
```mermaid
flowchart LR
    A[AkShare Public Data] --> B[Cleaning & Alignment]
    B --> C[Feature Engineering]
    C --> D[Nested Walk-Forward]
    D --> E[LambdaRank Forecast]
    E --> F[Portfolio Construction]
    F --> G[Strategy Evaluation]
    G --> H[Decision Dashboard]
```
        """
    )

    for s in steps:
        st.markdown(
            f'<div class="pipeline-step">'
            f"<strong>Step {s['step']} · {s['name']}</strong> — {s['label']}<br>"
            f"<span>{s['detail']}</span><br>"
            f"<span>Output: <code>{s['output']}</code></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Refresh Forecast")
    st.code("python run_phase1.py\npython run_ml.py\npython run_dashboard.py", language="bash")

    fc = dl.load_live_forecast()
    if fc:
        st.caption(f"Latest forecast: {fc.get('forecast_month')} · Updated {fc.get('updated_at')}")


# ── Main ──────────────────────────────────────────────────────────────────────

PAGES = {
    "Executive Overview": page_executive,
    "Market Environment": page_market,
    "Sector Intelligence": page_sector,
    "Explainable AI": page_xai,
    "Portfolio Performance": page_performance,
    "Model Evaluation": page_model_eval,
    "Data Pipeline": page_pipeline,
}


def main():
    st.sidebar.markdown("### Sector Rotation")
    st.sidebar.caption("Decision Support Platform")

    fc = dl.load_live_forecast()
    if fc:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Forecast**  \n{fc.get('forecast_month')}")
        st.sidebar.markdown(f"**As-Of**  \n{fc.get('data_as_of')}")
        if fc.get("top_holdings"):
            top3 = " · ".join(h["industry"] for h in fc["top_holdings"][:3])
            st.sidebar.markdown(f"**Top-3**  \n{top3}")

    page = st.sidebar.radio("Navigation", list(PAGES.keys()))

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Reload re-reads CSV files under `data/processed/`. "
        "It does not run `run_ml.py` automatically."
    )
    if st.sidebar.button("Reload Data", type="primary", use_container_width=True, key="reload_data"):
        st.cache_data.clear()
        st.session_state["last_reload_at"] = pd.Timestamp.now().strftime("%H:%M:%S")
        st.rerun()

    if st.session_state.get("last_reload_at"):
        st.sidebar.success(f"Data reloaded · {st.session_state['last_reload_at']}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Research prototype · Not investment advice")

    PAGES[page]()


if __name__ == "__main__":
    main()
