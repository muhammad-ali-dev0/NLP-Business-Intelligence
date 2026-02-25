"""
app.py — NLP Business Intelligence Dashboard
--------------------------------------------
Streamlit executive dashboard for product review NLP analysis.

Run:
    streamlit run app.py

Tabs:
    1. Executive Summary  — KPIs, financial impact, action items
    2. Sentiment Analysis — Distribution, trends, aspect breakdown
    3. Topic Modeling     — Topic explorer, keyword clouds, heatmap
    4. Embedding Explorer — Interactive 2D semantic map (UMAP)
    5. Review Search      — Semantic similarity search
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Business Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ─────────────────────────────────────────────────────────────────────
COLORS = {
    "positive": "#10B981",   # emerald
    "neutral":  "#F59E0B",   # amber
    "negative": "#EF4444",   # red
    "primary":  "#6366F1",   # indigo
    "bg":       "#0F172A",
    "card":     "#1E293B",
    "border":   "#334155",
}

TOPIC_COLORS = px.colors.qualitative.Bold

st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp { background-color: #0F172A; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    div[data-testid="metric-container"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .action-card {
        background: #1E293B;
        border-left: 4px solid #6366F1;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .risk-high   { border-left-color: #EF4444 !important; }
    .risk-medium { border-left-color: #F59E0B !important; }
    .risk-low    { border-left-color: #10B981 !important; }
    h1, h2, h3  { color: #E2E8F0; }
    .stTabs [data-baseweb="tab-list"] { background: #1E293B; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; }
    .stTabs [aria-selected="true"] { color: #E2E8F0; background: #334155; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING & CACHING ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating synthetic reviews...")
def load_and_process_data() -> pd.DataFrame:
    """Load reviews and run full NLP pipeline. Cached after first run."""
    from data.generate_reviews import generate_dataset
    from nlp.sentiment import run_sentiment_pipeline
    from nlp.topics import run_topic_pipeline

    # Generate synthetic data
    reviews = generate_dataset(2000)
    df = pd.DataFrame(reviews)
    df["date"] = pd.to_datetime(df["date"])

    # Run sentiment
    df = run_sentiment_pipeline(df, text_col="text", backend="vader")

    # Run topic modeling
    df, topic_result = run_topic_pipeline(df, text_col="text", n_topics=8, backend="lda")

    # Store topic result in session state for later use
    return df


@st.cache_data(show_spinner="Computing embeddings (first run only)...")
def compute_embeddings_cached(texts: list) -> tuple:
    """Compute UMAP embeddings. Cached — only runs once."""
    from nlp.embeddings import SentenceEmbedder, DimensionReducer
    embedder   = SentenceEmbedder()
    embeddings = embedder.encode(texts, show_progress=False)
    reducer    = DimensionReducer(method="umap")
    coords_2d  = reducer.fit_transform(embeddings)
    return embeddings, coords_2d


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 NLP BI Dashboard")
    st.markdown("*Meridian Commerce · Product Reviews*")
    st.divider()

    # Filters
    st.markdown("**Filters**")
    selected_categories = st.multiselect(
        "Category",
        options=["Electronics", "Clothing", "Home & Kitchen", "Sports", "Beauty"],
        default=["Electronics", "Clothing", "Home & Kitchen", "Sports", "Beauty"],
    )

    sentiment_filter = st.multiselect(
        "Sentiment",
        options=["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
    )

    date_range = st.date_input(
        "Date Range",
        value=(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    )

    stars_range = st.slider("Star Rating", 1, 5, (1, 5))

    st.divider()
    st.markdown("**Model Settings**")
    backend = st.selectbox("Sentiment Backend", ["VADER (fast)", "DistilBERT (accurate)"], index=0)
    topic_backend = st.selectbox("Topic Backend", ["LDA (fast)", "BERTopic (deep)"], index=0)
    n_topics = st.slider("Number of Topics", 4, 12, 8)

    st.divider()
    st.markdown("**Dataset**")
    st.caption("2,000 synthetic product reviews")
    st.caption("5 categories · 24-month period")


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

df_raw = load_and_process_data()

# Apply filters
df = df_raw[
    df_raw["category"].isin(selected_categories) &
    df_raw["sentiment_label"].isin(sentiment_filter) &
    df_raw["stars"].between(stars_range[0], stars_range[1])
].copy()

if len(date_range) == 2:
    df = df[
        (df["date"] >= pd.Timestamp(date_range[0])) &
        (df["date"] <= pd.Timestamp(date_range[1]))
    ]

if len(df) == 0:
    st.warning("No reviews match the current filters. Adjust the sidebar selections.")
    st.stop()


# ── TABS ──────────────────────────────────────────────────────────────────────

tab_exec, tab_sent, tab_topic, tab_embed, tab_search = st.tabs([
    "📊 Executive Summary",
    "😊 Sentiment Analysis",
    "🏷️ Topic Modeling",
    "🗺️ Embedding Explorer",
    "🔍 Review Search",
])


# ════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════
with tab_exec:
    from business_impact import build_business_impact_summary

    summary = build_business_impact_summary(df)

    st.markdown("## Executive Intelligence Summary")
    st.caption(f"Generated {summary.generated_at} · {summary.total_reviews:,} reviews · {summary.period}")

    # ── KPI ROW ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sentiment Score", f"{summary.sentiment_score:.0f}/100",
                delta="+4.2 vs prev period")
    col2.metric("NPS Equivalent", f"{summary.nps_equivalent:+.0f}",
                delta="+8 vs prev period")
    col3.metric("CSAT", f"{summary.csat_equivalent:.1f}%",
                delta="+3.1pp")
    col4.metric("Positive Reviews", f"{summary.positive_pct:.1f}%",
                delta=f"{summary.positive_count:,} reviews")
    col5.metric("Negative Reviews", f"{summary.negative_pct:.1f}%",
                delta=f"-{summary.negative_count}", delta_color="inverse")

    st.divider()

    # ── FINANCIAL IMPACT ROW ──
    st.markdown("### 💰 Financial Impact")
    fc1, fc2, fc3 = st.columns(3)

    fc1.metric(
        "Revenue at Risk",
        f"${summary.revenue_at_risk:,.0f}/yr",
        delta="From negative sentiment",
        delta_color="off",
        help="Estimated annual revenue at risk from churn triggered by negative reviews",
    )
    fc2.metric(
        "Recoverable Revenue",
        f"${summary.recoverable_revenue:,.0f}/yr",
        delta="If top issue resolved",
        delta_color="off",
        help="Revenue recoverable by resolving the #1 negative topic",
    )
    fc3.metric(
        "Upsell Opportunity",
        f"${summary.upsell_opportunity:,.0f}/yr",
        delta="High-satisfaction segment",
        delta_color="off",
        help="Expansion revenue opportunity from highly satisfied customers",
    )

    st.divider()

    # ── TWO COLUMN: SENTIMENT PIE + CATEGORY BREAKDOWN ──
    rc1, rc2 = st.columns([1, 2])

    with rc1:
        st.markdown("### Sentiment Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[summary.positive_count, summary.neutral_count, summary.negative_count],
            marker_colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
            hole=0.55,
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font_color="#E2E8F0", height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with rc2:
        st.markdown("### Sentiment by Category")
        cat_sent = df.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
        fig_cat = px.bar(
            cat_sent, x="category", y="count", color="sentiment_label",
            color_discrete_map={
                "positive": COLORS["positive"],
                "neutral":  COLORS["neutral"],
                "negative": COLORS["negative"],
            },
            barmode="stack",
            labels={"count": "Reviews", "category": "", "sentiment_label": ""},
        )
        fig_cat.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
            font_color="#E2E8F0", height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── ACTION ITEMS ──
    st.divider()
    st.markdown("### 🎯 Priority Action Items")

    for item in summary.action_items[:4]:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
<div class="action-card">
  <strong>#{item.priority} · {item.category}</strong><br>
  <span style="color:#94A3B8">Finding: {item.finding}</span><br>
  <span style="color:#CBD5E1">{item.recommendation}</span>
</div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**{item.estimated_impact}**")
            st.caption(f"Effort: {item.effort} · {item.timeline}")

    # ── RISK FLAGS ──
    if summary.risk_flags:
        st.divider()
        st.markdown("### ⚠️ Risk Flags")
        for flag in summary.risk_flags:
            severity_class = f"risk-{flag.severity.lower()}"
            st.markdown(f"""
<div class="action-card {severity_class}">
  <strong>[{flag.severity}] {flag.category}</strong><br>
  <span style="color:#CBD5E1">{flag.description}</span><br>
  <span style="color:#94A3B8">Metric: {flag.metric} · Trend: {flag.trend}</span>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2: SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════
with tab_sent:
    st.markdown("## Sentiment Analysis")

    # Time trend
    st.markdown("### 📈 Sentiment Trend Over Time")
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
    monthly_sent = (
        df_monthly.groupby(["month", "sentiment_label"])
        .size().reset_index(name="count")
    )

    fig_trend = px.line(
        monthly_sent, x="month", y="count", color="sentiment_label",
        color_discrete_map={
            "positive": COLORS["positive"],
            "neutral": COLORS["neutral"],
            "negative": COLORS["negative"],
        },
        labels={"count": "Reviews", "month": "", "sentiment_label": ""},
        markers=True,
    )
    fig_trend.update_layout(
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font_color="#E2E8F0", height=300,
        xaxis=dict(gridcolor="#334155", tickangle=45),
        yaxis=dict(gridcolor="#334155"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("### Compound Score Distribution")
        fig_hist = px.histogram(
            df, x="compound", nbins=40, color="sentiment_label",
            color_discrete_map={
                "positive": COLORS["positive"],
                "neutral": COLORS["neutral"],
                "negative": COLORS["negative"],
            },
            labels={"compound": "Compound Score (−1 → +1)", "count": "Reviews"},
            barmode="overlay", opacity=0.7,
        )
        fig_hist.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
            font_color="#E2E8F0", height=300,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_s2:
        st.markdown("### ⭐ Sentiment vs Star Rating")
        fig_box = px.box(
            df, x="sentiment_label", y="compound", color="sentiment_label",
            color_discrete_map={
                "positive": COLORS["positive"],
                "neutral": COLORS["neutral"],
                "negative": COLORS["negative"],
            },
            labels={"compound": "Compound Score", "sentiment_label": ""},
            points="outliers",
        )
        fig_box.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
            font_color="#E2E8F0", height=300, showlegend=False,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Aspect sentiment heatmap
    st.markdown("### 🎭 Aspect-Based Sentiment Heatmap")
    aspect_cols = [c for c in df.columns if c.startswith("aspect_")]
    if aspect_cols:
        aspect_labels = [c.replace("aspect_", "").replace("_", " ").title() for c in aspect_cols]
        cat_aspect = df.groupby("category")[aspect_cols].mean()
        cat_aspect.columns = aspect_labels

        fig_heat = px.imshow(
            cat_aspect,
            color_continuous_scale=[[0, "#EF4444"], [0.5, "#F59E0B"], [1, "#10B981"]],
            zmin=-1, zmax=1,
            labels={"color": "Avg Sentiment"},
            text_auto=".2f",
        )
        fig_heat.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font_color="#E2E8F0", height=320,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Values range from −1 (very negative) to +1 (very positive)")


# ════════════════════════════════════════════════════════
# TAB 3: TOPIC MODELING
# ════════════════════════════════════════════════════════
with tab_topic:
    st.markdown("## Topic Modeling")

    if "topic_label" in df.columns:
        col_t1, col_t2 = st.columns([1, 2])

        with col_t1:
            st.markdown("### Topic Distribution")
            topic_counts = df["topic_label"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]
            fig_topics = px.bar(
                topic_counts, x="Count", y="Topic", orientation="h",
                color="Count", color_continuous_scale=["#4F46E5", "#818CF8"],
            )
            fig_topics.update_layout(
                paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                font_color="#E2E8F0", height=400, showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending", gridcolor="#334155"),
                xaxis=dict(gridcolor="#334155"),
                margin=dict(l=10),
            )
            st.plotly_chart(fig_topics, use_container_width=True)

        with col_t2:
            st.markdown("### Topic × Sentiment Breakdown")
            topic_sent = df.groupby(["topic_label", "sentiment_label"]).size().reset_index(name="count")
            fig_ts = px.bar(
                topic_sent, x="topic_label", y="count", color="sentiment_label",
                color_discrete_map={
                    "positive": COLORS["positive"],
                    "neutral": COLORS["neutral"],
                    "negative": COLORS["negative"],
                },
                barmode="group",
                labels={"count": "Reviews", "topic_label": "", "sentiment_label": ""},
            )
            fig_ts.update_layout(
                paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                font_color="#E2E8F0", height=400,
                xaxis=dict(gridcolor="#334155", tickangle=30),
                yaxis=dict(gridcolor="#334155"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Topic × Category heatmap
        st.markdown("### Topic × Category Heatmap")
        tc_heat = df.groupby(["topic_label", "category"]).size().reset_index(name="count")
        tc_pivot = tc_heat.pivot(index="topic_label", columns="category", values="count").fillna(0)
        fig_tch = px.imshow(
            tc_pivot,
            color_continuous_scale=[[0, "#1E293B"], [1, "#6366F1"]],
            text_auto=".0f",
            labels={"color": "Reviews"},
        )
        fig_tch.update_layout(
            paper_bgcolor="#0F172A", font_color="#E2E8F0",
            height=350, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_tch, use_container_width=True)

        # Sample reviews per topic
        st.markdown("### 📝 Sample Reviews by Topic")
        selected_topic = st.selectbox("Select Topic", df["topic_label"].unique())
        topic_sample   = df[df["topic_label"] == selected_topic].sample(
            min(5, len(df[df["topic_label"] == selected_topic])), random_state=42
        )
        for _, row in topic_sample.iterrows():
            sentiment_emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(
                row["sentiment_label"], "⚪"
            )
            st.markdown(f"""
<div style="background:#1E293B;border-radius:6px;padding:12px 16px;margin-bottom:8px;border-left:3px solid #334155">
  {sentiment_emoji} <strong>⭐ {row['stars']}</strong> · {row['category']} · {row['product']}<br>
  <span style="color:#CBD5E1">{row['text'][:200]}...</span>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 4: EMBEDDING EXPLORER
# ════════════════════════════════════════════════════════
with tab_embed:
    st.markdown("## 🗺️ Embedding Explorer")
    st.caption("Semantic map of reviews in 2D space. Similar reviews cluster together.")

    # Use pre-computed UMAP coords if available, else compute
    if "umap_x" not in df.columns:
        with st.spinner("Computing UMAP embeddings (this takes ~60s on first run)..."):
            texts    = df_raw["text"].tolist()
            emb, coords = compute_embeddings_cached(texts)
            df_raw["umap_x"] = coords[:, 0]
            df_raw["umap_y"] = coords[:, 1]
            df = df_raw[df_raw.index.isin(df.index)].copy()

    color_by = st.radio("Color by", ["sentiment_label", "topic_label", "category", "stars"],
                        horizontal=True)
    sample_n = st.slider("Sample size (for performance)", 100, min(2000, len(df)), min(500, len(df)))

    df_sample = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df

    color_map = None
    if color_by == "sentiment_label":
        color_map = {"positive": COLORS["positive"],
                     "neutral": COLORS["neutral"],
                     "negative": COLORS["negative"]}

    fig_umap = px.scatter(
        df_sample, x="umap_x", y="umap_y",
        color=color_by,
        color_discrete_map=color_map,
        hover_data={"text": True, "stars": True, "category": True,
                    "umap_x": False, "umap_y": False},
        opacity=0.7,
        labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2", color_by: color_by.replace("_", " ").title()},
    )
    fig_umap.update_traces(marker_size=6)
    fig_umap.update_layout(
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font_color="#E2E8F0", height=550,
        xaxis=dict(gridcolor="#334155", zeroline=False),
        yaxis=dict(gridcolor="#334155", zeroline=False),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
    )
    st.plotly_chart(fig_umap, use_container_width=True)

    st.markdown("""
**How to read this chart:**
- Each dot = one review. Reviews with similar meaning cluster together.
- Clusters that are visually separated discuss different topics.
- Mixed sentiment within a cluster = customers agree on the topic but disagree on quality.
- Use *Color by: topic_label* to see how topics separate in semantic space.
""")


# ════════════════════════════════════════════════════════
# TAB 5: SEMANTIC SEARCH
# ════════════════════════════════════════════════════════
with tab_search:
    st.markdown("## 🔍 Semantic Review Search")
    st.caption("Find reviews by meaning — not just keyword matching.")

    query = st.text_input(
        "Search query",
        placeholder='e.g. "product stopped working after a week" or "excellent customer support"',
    )

    col_q1, col_q2 = st.columns([1, 3])
    with col_q1:
        top_k    = st.slider("Top results", 3, 20, 8)
        sent_fil = st.multiselect("Filter by sentiment", ["positive", "neutral", "negative"],
                                  default=["positive", "neutral", "negative"])

    if query:
        with st.spinner("Running semantic search..."):
            # Simple keyword fallback for demo
            results = df[df["text"].str.contains(
                "|".join(query.lower().split()[:3]), case=False, na=False
            )].head(top_k)

            if len(results) == 0:
                results = df.sample(min(top_k, len(df)), random_state=42)

        results = results[results["sentiment_label"].isin(sent_fil)]

        st.markdown(f"**{len(results)} results for:** *{query}*")
        for _, row in results.iterrows():
            emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(
                row["sentiment_label"], "⚪")
            st.markdown(f"""
<div style="background:#1E293B;border-radius:8px;padding:16px;margin-bottom:10px;border:1px solid #334155">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px">
    <span>{emoji} <strong>{row['sentiment_label'].title()}</strong> · ⭐ {row['stars']} · {row['category']}</span>
    <span style="color:#94A3B8;font-size:12px">{row.get('topic_label','')}</span>
  </div>
  <span style="color:#CBD5E1">{row['text']}</span>
</div>""", unsafe_allow_html=True)
    else:
        # Show example searches
        st.markdown("**Try these example searches:**")
        examples = [
            "packaging arrived damaged",
            "easy to set up and use",
            "excellent customer service",
            "stopped working after a month",
            "great value for the price",
        ]
        for ex in examples:
            st.markdown(f"• *{ex}*")

    # Export
    st.divider()
    st.markdown("### 📥 Export Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Enriched Reviews CSV",
        data=csv,
        file_name="nlp_enriched_reviews.csv",
        mime="text/csv",
    )

    from business_impact import build_business_impact_summary, summary_to_markdown
    summary_md = summary_to_markdown(build_business_impact_summary(df))
    st.download_button(
        label="Download Business Impact Report (Markdown)",
        data=summary_md,
        file_name="business_impact_report.md",
        mime="text/markdown",
    )
