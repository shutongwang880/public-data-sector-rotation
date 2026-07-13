"""Light institutional theme for Decision Support Dashboard."""

LIGHT_INSTITUTIONAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #f7f8fa;
    --card: #ffffff;
    --card2: #f0f3f8;
    --border: #d8dee9;
    --text: #1a2332;
    --muted: #5c6b7f;
    --accent: #1e5a8a;
    --green: #1a7f4b;
    --red: #c0392b;
}

/* 顶部工具栏留白，避免与标题挤压 */
.block-container {
    padding-top: 3.75rem !important;
    max-width: 1280px;
}

/* 仅隐藏 Deploy（云端部署入口），保留其余工具栏布局 */
[data-testid="stDeployButton"] {
    display: none !important;
}

/* 隐藏 Markdown ## 标题旁的锚点链接（# 图标，仅跳到本页顶部） */
[data-testid="stHeaderActionElements"],
a[data-testid="stHeaderActionButton"],
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600 !important; }
p, li, span, label { color: var(--muted); }

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--accent) !important;
    font-size: 0.95rem !important;
}

.page-title {
    color: var(--text);
    font-size: 1.65rem;
    font-weight: 700;
    margin: 2rem 0 0.35rem 0;
    line-height: 1.3;
}
.page-subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0 0 1.25rem 0;
}

.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0 1.25rem 0;
}
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    min-width: 10.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-label {
    color: var(--muted);
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.4rem;
    line-height: 1.3;
}
.kpi-val {
    color: var(--text);
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.4;
    white-space: normal;
    word-break: keep-all;
    overflow-wrap: break-word;
}

.hero-card {
    background: linear-gradient(135deg, #eef4fb 0%, #ffffff 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.hero-card h3 { color: var(--accent) !important; margin: 0 0 0.4rem 0; font-size: 0.85rem; }
.hero-card p { color: var(--text); margin: 0; font-size: 1.05rem; line-height: 1.6; }

.decision-summary {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.7;
}

.regime-pill {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    border: 1px solid var(--border);
    background: var(--card2);
    color: var(--accent);
}

.rank-overweight { color: var(--green); font-weight: 600; }
.rank-neutral { color: var(--muted); }
.rank-underweight { color: var(--red); font-weight: 600; }

.pipeline-step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.pipeline-step strong { color: var(--accent); }
.pipeline-step span { color: var(--muted); font-size: 0.85rem; }
</style>
"""

DARK_FINTECH_CSS = LIGHT_INSTITUTIONAL_CSS


def inject_theme() -> None:
    import streamlit as st

    st.markdown(LIGHT_INSTITUTIONAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Page title without Streamlit ## anchor-link icon."""
    import streamlit as st

    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str]]) -> None:
    """Render KPI cards with full-width values (fixes truncated Chinese dates)."""
    import streamlit as st

    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)
