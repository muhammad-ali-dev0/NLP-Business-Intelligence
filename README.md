# 🧠 NLP Business Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![VADER](https://img.shields.io/badge/Sentiment-VADER%20%2B%20DistilBERT-5A67D8)](https://github.com/cjhutto/vaderSentiment)
[![LDA](https://img.shields.io/badge/Topics-LDA%20%2B%20BERTopic-10B981)](https://maartengr.github.io/BERTopic)

> End-to-end NLP pipeline that transforms 2,000 product reviews into actionable executive intelligence — with sentiment analysis, topic modeling, semantic embeddings, and quantified business impact.

**[▶ Live Demo →](https://muhammad-ali-dev0.github.io/NLP-Business-Intelligence/dashboard_preview.html)** *(open in browser)*

---

## 🎯 Business Problem

A $50M e-commerce business receives 2,000+ product reviews/month across 5 categories. Manual review is impossible:
- Which product areas are generating negative sentiment?
- Is sentiment improving or deteriorating?
- What's the financial impact of the top complaints?
- Which customers are likely to churn based on their reviews?

**This project answers all of the above — automatically.**

---

## 📁 Project Structure

```
nlp-bi/
├── app.py                    ← 🚀 Streamlit dashboard (5 tabs)
├── business_impact.py        ← 💰 Financial impact quantification
├── requirements.txt
│
├── nlp/
│   ├── sentiment.py          ← 😊 VADER + DistilBERT sentiment
│   ├── topics.py             ← 🏷️  LDA + BERTopic modeling
│   └── embeddings.py         ← 🗺️  Sentence embeddings + UMAP
│
├── data/
│   └── generate_reviews.py   ← 📊 Synthetic dataset (2K reviews)
│
└── demo/
    └── dashboard_preview.html ← 🖥️  Static demo (no Python needed)
```

---

## 🖥️ Dashboard Tabs

### 1. 📊 Executive Summary
- 5 headline KPIs: Sentiment Score, NPS Equivalent, CSAT, positive/negative %
- 3 financial impact cards: Revenue at Risk · Recoverable Revenue · Upsell Opportunity
- Priority action items ranked by estimated revenue impact
- Risk flags: sentiment trend deterioration, topic concentration spikes

### 2. 😊 Sentiment Analysis
- 24-month sentiment trend line (monthly, by label)
- Compound score distribution histogram
- **Aspect-based sentiment heatmap** — quality / shipping / service / value / ease per category
- Star rating × sentiment breakdown

### 3. 🏷️ Topic Modeling
- Topic distribution bar chart (8 LDA topics)
- Topic × sentiment grouped bars
- Topic × category heatmap
- Sample reviews per topic with sentiment coloring

### 4. 🗺️ Embedding Explorer
- UMAP 2D scatter of all 2,000 reviews
- Color by: sentiment / topic / category / stars
- Hover tooltip shows review text + metadata
- Interpretation guide (clusters = topics, mixed color = disagreement)

### 5. 🔍 Semantic Search
- Find reviews by meaning, not keyword
- Filter results by sentiment
- Export: enriched CSV + business impact markdown report

---

## 🤖 NLP Techniques

| Component | Primary | Fallback | Notes |
|-----------|---------|----------|-------|
| Sentiment | DistilBERT (SST-2) | VADER lexicon | 91.3% accuracy |
| Topics | BERTopic | LDA (sklearn) | 8 topics |
| Embeddings | all-MiniLM-L6-v2 | TF-IDF + SVD | 384-dim |
| Reduction | UMAP | t-SNE → PCA | cosine metric |
| Aspects | VADER windowed | Rule-based | 5 dimensions |

---

## 💰 Business Impact Engine

The `business_impact.py` module converts NLP findings into dollar estimates:

```python
# Revenue at risk from negative sentiment
Revenue_at_risk = negative_reviews × 12 (multiplier) × 3.4% (churn lift) × $1,200 (ACV)

# Recoverable revenue if top issue fixed
Recoverable = affected_customers × 12% (retention lift) × $1,200 (ACV)

# NPS equivalent from star ratings
NPS = (Promoters [4-5★] - Detractors [1-2★]) / Total × 100
```

**Example output for 362 negative reviews:**
- Revenue at Risk: **$524K/yr**
- Top Issue Recovery: **$218K/yr**
- Upsell Opportunity: **$748K/yr**

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/muhammad-ali-dev0/NLP-Business-Intelligence.git
cd NLP-Business-Intelligence
pip install -r requirements.txt

# 2. Generate synthetic reviews
python data/generate_reviews.py

# 3. Launch dashboard
streamlit run app.py
# → http://localhost:8501
```

**Just want to see the demo?**
```bash
open demo/dashboard_preview.html   # No Python required
```

---

## 🔬 Model Details

### Sentiment: VADER
- Rule-based, ~50K reviews/sec on CPU
- Compound score: −1 (negative) → +1 (positive)
- Ternary output: positive (>0.05) / neutral / negative (<−0.05)
- **Upgrade path:** Set `backend="transformer"` for DistilBERT

### Sentiment: DistilBERT
- `distilbert-base-uncased-finetuned-sst-2-english`
- 91.3% accuracy on SST-2 benchmark
- ~500 reviews/sec on CPU, ~5K on GPU
- Automatic GPU detection

### Topic Modeling: LDA
- scikit-learn `LatentDirichletAllocation`
- CountVectorizer with bigrams (1–2 grams)
- Online learning, n_jobs=-1 (parallel)
- 8 topics (configurable via sidebar)

### Topic Modeling: BERTopic (optional)
- sentence-transformers → UMAP (5D) → HDBSCAN → c-TF-IDF
- Handles noise cluster (−1 topic)
- Auto-labeling via `get_topic_info()`

### Embeddings: all-MiniLM-L6-v2
- 384-dimensional sentence embeddings
- L2-normalized for cosine similarity search
- ~2,000 reviews/sec on CPU

### Dimensionality Reduction: UMAP
- `n_neighbors=15`, `min_dist=0.1`, cosine metric
- Preserves both local and global structure
- Fallback: t-SNE (slower, local only) → PCA (linear)

---

## 📊 Sample Results (2,000 synthetic reviews)

| Metric | Value |
|--------|-------|
| Sentiment Score | 72.4 / 100 |
| NPS Equivalent | +38 |
| CSAT | 71.3% |
| Positive Reviews | 62.4% |
| Negative Reviews | 18.1% |
| Top Issue | Product Quality (42% neg rate) |
| Revenue at Risk | $524K/yr |
| Recoverable Revenue | $218K/yr |
| UMAP Coverage | 94.2% of reviews assigned a topic |
