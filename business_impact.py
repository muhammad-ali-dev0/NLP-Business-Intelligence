"""
business_impact.py
------------------
Converts NLP analysis results into executive-ready business insights.

Produces:
  - Financial impact estimates from sentiment trends
  - Priority action items ranked by estimated revenue impact
  - Risk flags (sentiment declining, topic spikes)
  - Exportable PDF/markdown summary
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


# ── DATA MODELS ───────────────────────────────────────────────────────────────

@dataclass
class ActionItem:
    priority:        int          # 1 = highest
    category:        str
    finding:         str
    recommendation:  str
    estimated_impact: str        # e.g. "+$420K ARR"
    effort:          str         # Low / Medium / High
    timeline:        str         # e.g. "2–4 weeks"


@dataclass
class RiskFlag:
    severity:    str             # Critical / High / Medium / Low
    category:    str
    description: str
    metric:      str
    trend:       str             # direction / magnitude


@dataclass
class BusinessImpactSummary:
    generated_at:     str
    period:           str
    total_reviews:    int
    sentiment_score:  float      # weighted NPS-like score
    nps_equivalent:   float
    csat_equivalent:  float

    # Volume
    positive_count:   int
    neutral_count:    int
    negative_count:   int
    positive_pct:     float
    negative_pct:     float

    # Financial estimates
    revenue_at_risk:  float      # $ from negative sentiment
    recoverable_revenue: float   # $ if top issue resolved
    upsell_opportunity: float    # $ from high-satisfaction segment

    # Insights
    top_issue:        str
    top_strength:     str
    trending_topic:   str

    action_items:     list[ActionItem] = field(default_factory=list)
    risk_flags:       list[RiskFlag]   = field(default_factory=list)


# ── FINANCIAL ESTIMATIONS ─────────────────────────────────────────────────────

# Industry benchmarks used for estimation
BENCHMARKS = {
    "avg_customer_value_annual":  1_200,   # $1,200 avg annual spend
    "churn_lift_per_neg_review":  0.034,   # 3.4% higher churn per negative review
    "conversion_lift_per_pos":    0.018,   # 1.8% higher conversion from positive
    "review_multiplier":          12,      # 1 review = 12 customers' experience
    "resolution_retention_lift":  0.12,    # 12% retention improvement if top issue fixed
}


def estimate_revenue_at_risk(negative_count: int, avg_customer_value: float = 1_200) -> float:
    """
    Estimate annual revenue at risk from negative reviews.
    Based on: each negative review represents ~12 affected customers,
    with 3.4% incremental churn probability.
    """
    affected_customers = negative_count * BENCHMARKS["review_multiplier"]
    churned_customers  = affected_customers * BENCHMARKS["churn_lift_per_neg_review"]
    return round(churned_customers * avg_customer_value, 0)


def estimate_recoverable_revenue(df: pd.DataFrame, top_topic: str,
                                 avg_customer_value: float = 1_200) -> float:
    """
    If the top negative topic is resolved, estimate retained revenue.
    """
    topic_negative = df[
        (df.get("topic_label", pd.Series(dtype=str)) == top_topic) &
        (df.get("sentiment_label", pd.Series(dtype=str)) == "negative")
    ]
    n = len(topic_negative)
    affected = n * BENCHMARKS["review_multiplier"]
    retained = affected * BENCHMARKS["resolution_retention_lift"]
    return round(retained * avg_customer_value, 0)


def compute_nps_equivalent(df: pd.DataFrame) -> float:
    """
    NPS-equivalent from sentiment scores.
    5-star reviews → Promoters, 1–2 star → Detractors.
    """
    if "stars" not in df.columns:
        # Fall back to sentiment
        promoters  = (df["sentiment_label"] == "positive").sum()
        detractors = (df["sentiment_label"] == "negative").sum()
        total = len(df)
    else:
        promoters  = (df["stars"] >= 4).sum()
        detractors = (df["stars"] <= 2).sum()
        total      = len(df)

    if total == 0:
        return 0.0
    return round((promoters - detractors) / total * 100, 1)


def compute_csat(df: pd.DataFrame) -> float:
    """CSAT = % of positive/4-5 star responses."""
    if "stars" in df.columns:
        return round((df["stars"] >= 4).sum() / len(df) * 100, 1)
    return round((df["sentiment_label"] == "positive").sum() / len(df) * 100, 1)


# ── ACTION ITEM GENERATION ────────────────────────────────────────────────────

def generate_action_items(df: pd.DataFrame, topic_col: str = "topic_label") -> list[ActionItem]:
    """
    Generate prioritized action items from NLP findings.
    """
    actions = []
    priority = 1

    if topic_col not in df.columns or "sentiment_label" not in df.columns:
        return actions

    # Find worst topics by negative sentiment rate
    topic_sentiment = df.groupby(topic_col).agg(
        total=("sentiment_label", "count"),
        negative=(  "sentiment_label", lambda x: (x == "negative").sum()),
        avg_stars=("stars", "mean") if "stars" in df.columns else ("sentiment_label", "count"),
    ).reset_index()

    topic_sentiment["neg_rate"] = topic_sentiment["negative"] / topic_sentiment["total"]
    topic_sentiment = topic_sentiment.sort_values("neg_rate", ascending=False)

    for _, row in topic_sentiment.head(4).iterrows():
        topic    = row[topic_col]
        neg_rate = row["neg_rate"]
        neg_n    = row["negative"]

        if neg_rate > 0.35:
            estimated_impact = f"+${estimate_recoverable_revenue(df, topic):,.0f}/yr"
            actions.append(ActionItem(
                priority=priority,
                category=topic,
                finding=f"{neg_rate:.0%} of {topic} reviews are negative ({neg_n} reviews)",
                recommendation=_get_recommendation(topic),
                estimated_impact=estimated_impact,
                effort="Medium",
                timeline="4–8 weeks",
            ))
            priority += 1

    # Add upsell opportunity from top positive topic
    top_positive_topic = (
        df[df["sentiment_label"] == "positive"]
        .groupby(topic_col)
        .size()
        .idxmax()
        if len(df[df["sentiment_label"] == "positive"]) > 0 else "Unknown"
    )

    if top_positive_topic != "Unknown":
        actions.append(ActionItem(
            priority=priority,
            category=top_positive_topic,
            finding=f"Customers consistently praise {top_positive_topic}",
            recommendation=f"Feature {top_positive_topic} prominently in marketing. "
                           f"Upsell loyal customers with highest satisfaction.",
            estimated_impact="+12% upsell conversion",
            effort="Low",
            timeline="2–3 weeks",
        ))

    return actions


def _get_recommendation(topic: str) -> str:
    recommendations = {
        "Product Quality":    "Audit top-returned SKUs. Partner with QA team on material specs. "
                              "Add 30-day quality guarantee to reduce purchase friction.",
        "Shipping & Delivery": "Negotiate SLA with logistics partners. Set up proactive delay notifications. "
                               "Consider regional warehouse for top markets.",
        "Customer Service":   "Implement AI-assisted triage for common issues. "
                              "Target first-response time < 4 hours. Train team on top complaint categories.",
        "Value for Money":    "Add value-tier product option at 30% lower price point. "
                              "Bundle slow-moving complementary items to improve perceived value.",
        "Ease of Use":        "Create video onboarding for top 3 use cases. "
                              "Redesign setup flow based on user drop-off points.",
        "Product Defects":    "Root-cause analysis on top defect categories. "
                              "Pre-shipment QC check for flagged SKUs. Update returns process.",
    }
    for key, rec in recommendations.items():
        if key.lower() in topic.lower():
            return rec
    return f"Investigate root cause of negative {topic} reviews. Survey customers who left 1-2 star reviews."


# ── RISK FLAGS ────────────────────────────────────────────────────────────────

def detect_risk_flags(df: pd.DataFrame, date_col: str = "date") -> list[RiskFlag]:
    flags = []

    if date_col in df.columns and "sentiment_label" in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted    = df.sort_values(date_col)

        # Split into halves and compare
        mid = len(df_sorted) // 2
        early_neg = (df_sorted.iloc[:mid]["sentiment_label"] == "negative").mean()
        late_neg  = (df_sorted.iloc[mid:]["sentiment_label"] == "negative").mean()

        delta = late_neg - early_neg
        if delta > 0.05:
            flags.append(RiskFlag(
                severity="High",
                category="Sentiment Trend",
                description=f"Negative sentiment rate increased {delta:.1%} in recent period",
                metric=f"{late_neg:.1%} vs {early_neg:.1%} earlier",
                trend=f"+{delta:.1%}",
            ))
        elif delta < -0.05:
            flags.append(RiskFlag(
                severity="Low",
                category="Sentiment Trend",
                description=f"Negative sentiment improving — down {abs(delta):.1%}",
                metric=f"{late_neg:.1%} vs {early_neg:.1%} earlier",
                trend=f"{delta:.1%}",
            ))

    # Check for topic spikes
    if "topic_label" in df.columns and date_col in df.columns:
        recent = df[df[date_col] >= df[date_col].max() - pd.Timedelta(days=30)]
        topic_counts = recent["topic_label"].value_counts(normalize=True)
        for topic, pct in topic_counts.items():
            if pct > 0.40:
                flags.append(RiskFlag(
                    severity="Medium",
                    category="Topic Concentration",
                    description=f"{topic} is dominating recent reviews ({pct:.0%} of last 30 days)",
                    metric=f"{pct:.0%} of recent reviews",
                    trend="Spike",
                ))
                break

    return flags


# ── MAIN SUMMARY BUILDER ──────────────────────────────────────────────────────

def build_business_impact_summary(df: pd.DataFrame) -> BusinessImpactSummary:
    """
    Build complete BusinessImpactSummary from enriched review DataFrame.
    Expects columns: sentiment_label, compound, topic_label, stars (optional), date.
    """
    total = len(df)
    pos   = (df["sentiment_label"] == "positive").sum()
    neu   = (df["sentiment_label"] == "neutral").sum()
    neg   = (df["sentiment_label"] == "negative").sum()

    # Find top topic overall (most common)
    top_topic = (
        df["topic_label"].mode()[0]
        if "topic_label" in df.columns and len(df) > 0
        else "N/A"
    )

    # Find top negative topic
    neg_df = df[df["sentiment_label"] == "negative"]
    top_neg_topic = (
        neg_df["topic_label"].mode()[0]
        if "topic_label" in neg_df.columns and len(neg_df) > 0
        else "Unknown"
    )

    # Find top strength (positive topic)
    pos_df = df[df["sentiment_label"] == "positive"]
    top_strength = (
        pos_df["topic_label"].mode()[0]
        if "topic_label" in pos_df.columns and len(pos_df) > 0
        else "Unknown"
    )

    revenue_at_risk     = estimate_revenue_at_risk(neg)
    recoverable         = estimate_recoverable_revenue(df, top_neg_topic)
    upsell_opportunity  = round(pos * 0.05 * BENCHMARKS["avg_customer_value_annual"], 0)

    # Weighted sentiment score: (pos - neg) / total, scaled to 0–100
    sentiment_score = round(((pos - neg) / total) * 50 + 50, 1)

    action_items = generate_action_items(df)
    risk_flags   = detect_risk_flags(df)

    return BusinessImpactSummary(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        period="Trailing 24 Months",
        total_reviews=total,
        sentiment_score=sentiment_score,
        nps_equivalent=compute_nps_equivalent(df),
        csat_equivalent=compute_csat(df),
        positive_count=int(pos),
        neutral_count=int(neu),
        negative_count=int(neg),
        positive_pct=round(pos / total * 100, 1),
        negative_pct=round(neg / total * 100, 1),
        revenue_at_risk=revenue_at_risk,
        recoverable_revenue=recoverable,
        upsell_opportunity=upsell_opportunity,
        top_issue=top_neg_topic,
        top_strength=top_strength,
        trending_topic=top_topic,
        action_items=action_items,
        risk_flags=risk_flags,
    )


def summary_to_markdown(summary: BusinessImpactSummary) -> str:
    """Export summary as markdown for reports."""
    lines = [
        f"# NLP Business Impact Report",
        f"**Generated:** {summary.generated_at} | **Period:** {summary.period}",
        "",
        "## 📊 Headline Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Reviews Analyzed | {summary.total_reviews:,} |",
        f"| Sentiment Score | {summary.sentiment_score}/100 |",
        f"| NPS Equivalent | {summary.nps_equivalent:+.1f} |",
        f"| CSAT Equivalent | {summary.csat_equivalent:.1f}% |",
        f"| Positive Reviews | {summary.positive_pct:.1f}% ({summary.positive_count:,}) |",
        f"| Negative Reviews | {summary.negative_pct:.1f}% ({summary.negative_count:,}) |",
        "",
        "## 💰 Financial Impact",
        f"| | Amount |",
        f"|---|---|",
        f"| Revenue at Risk (from negative sentiment) | ${summary.revenue_at_risk:,.0f}/yr |",
        f"| Recoverable Revenue (if top issue fixed) | ${summary.recoverable_revenue:,.0f}/yr |",
        f"| Upsell Opportunity (high-satisfaction) | ${summary.upsell_opportunity:,.0f}/yr |",
        "",
        "## 🎯 Priority Action Items",
    ]

    for item in summary.action_items:
        lines += [
            f"### {item.priority}. {item.category}",
            f"**Finding:** {item.finding}",
            f"**Action:** {item.recommendation}",
            f"**Impact:** {item.estimated_impact} | **Effort:** {item.effort} | **Timeline:** {item.timeline}",
            "",
        ]

    if summary.risk_flags:
        lines += ["## ⚠️ Risk Flags", ""]
        for flag in summary.risk_flags:
            lines += [
                f"**[{flag.severity}] {flag.category}:** {flag.description}",
                f"*Metric: {flag.metric} | Trend: {flag.trend}*",
                "",
            ]

    return "\n".join(lines)
