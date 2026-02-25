"""
nlp/topics.py
-------------
Topic modeling pipeline supporting:
  1. BERTopic — transformer embeddings + HDBSCAN + c-TF-IDF (state-of-art)
  2. LDA (Latent Dirichlet Allocation) — classical, interpretable, fast

Both backends return a unified TopicResult format compatible with the dashboard.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")


# ── DATA MODEL ─────────────────────────────────────────────────────────────────

@dataclass
class Topic:
    id:           int
    label:        str               # human-readable name
    keywords:     list[tuple[str, float]]   # (word, weight) top 10
    size:         int               # number of documents
    sentiment_avg: float            # avg compound score for docs in this topic
    coherence:    float             # topic coherence score (0–1)


@dataclass
class TopicModelResult:
    topics:       list[Topic]
    doc_topics:   list[int]         # topic assignment per document
    doc_probs:    list[float]       # topic probability per document
    model_type:   str               # "bertopic" or "lda"
    n_topics:     int
    coverage:     float             # % of docs assigned a topic (not noise)


# ── TEXT PREPROCESSING ─────────────────────────────────────────────────────────

import re
import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "this",
    "that", "i", "my", "me", "we", "you", "your", "they", "their", "have",
    "had", "has", "will", "would", "could", "should", "not", "no", "just",
    "so", "very", "also", "about", "more", "than", "its", "our", "are",
    "do", "did", "been", "when", "which", "who", "what", "how", "all",
    "if", "can", "get", "got", "one", "two", "three", "bought", "buy",
    "use", "used", "using", "product", "item", "review", "order", "ordered",
}


def preprocess(text: str) -> list[str]:
    """Tokenize and clean text for topic modeling."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


# ── LDA BACKEND ───────────────────────────────────────────────────────────────

class LDATopicModel:
    """
    Classical LDA using scikit-learn.
    Interpretable, fast, no GPU required.
    Works well for 500–50K documents.
    """

    def __init__(self, n_topics: int = 8, max_iter: int = 30, random_state: int = 42):
        self.n_topics     = n_topics
        self.max_iter     = max_iter
        self.random_state = random_state
        self.model        = None
        self.vectorizer   = None

    def fit(self, texts: list[str]) -> "LDATopicModel":
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        processed = [" ".join(preprocess(t)) for t in texts]

        self.vectorizer = CountVectorizer(
            max_df=0.95, min_df=3,
            max_features=5000,
            ngram_range=(1, 2),
        )
        dtm = self.vectorizer.fit_transform(processed)

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method="online",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(dtm)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        processed = [" ".join(preprocess(t)) for t in texts]
        dtm = self.vectorizer.transform(processed)
        return self.model.transform(dtm)

    def get_topic_keywords(self, n_words: int = 10) -> list[list[tuple[str, float]]]:
        vocab = self.vectorizer.get_feature_names_out()
        keywords = []
        for topic_weights in self.model.components_:
            top_ids = topic_weights.argsort()[-n_words:][::-1]
            top_words = [(vocab[i], round(topic_weights[i] / topic_weights.sum(), 4))
                         for i in top_ids]
            keywords.append(top_words)
        return keywords

    def auto_label(self, keywords: list[tuple[str, float]]) -> str:
        """Heuristic topic labeling from top keywords."""
        top_words = [w for w, _ in keywords[:3]]
        word_str = " ".join(top_words)

        label_map = {
            ("quality", "build", "material", "durable"): "Product Quality",
            ("shipping", "delivery", "arrived", "package"): "Shipping & Delivery",
            ("service", "support", "return", "refund"): "Customer Service",
            ("price", "value", "worth", "money", "cost"): "Value for Money",
            ("easy", "simple", "setup", "instructions"): "Ease of Use",
            ("love", "great", "amazing", "excellent"): "Overall Positive",
            ("broke", "broken", "stopped", "defective"): "Product Defects",
            ("size", "fit", "color", "style"): "Product Fit & Style",
        }
        for keywords_check, label in label_map.items():
            if any(k in word_str for k in keywords_check):
                return label
        return f"Topic: {top_words[0].title()}"


# ── BERTOPIC BACKEND ──────────────────────────────────────────────────────────

class BERTopicModel:
    """
    BERTopic: sentence embeddings → UMAP → HDBSCAN → c-TF-IDF.
    State-of-the-art topic coherence. Requires bertopic + sentence-transformers.

    Falls back to LDA if not available.
    """

    def __init__(self, min_topic_size: int = 20, n_neighbors: int = 15):
        self.min_topic_size = min_topic_size
        self.n_neighbors    = n_neighbors
        self.model          = None
        self.available      = False
        self._try_load()

    def _try_load(self):
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN

            umap_model  = UMAP(n_neighbors=self.n_neighbors, n_components=5,
                               metric="cosine", random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=self.min_topic_size,
                                    metric="euclidean", prediction_data=True)
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            self.model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                embedding_model=embedding_model,
                verbose=False,
            )
            self.available = True
            print("✅ BERTopic loaded with all-MiniLM-L6-v2 embeddings")
        except ImportError as e:
            print(f"⚠️  BERTopic not available ({e}), will use LDA fallback.")

    def fit_transform(self, texts: list[str]) -> tuple[list[int], np.ndarray]:
        if not self.available:
            raise RuntimeError("BERTopic not available")
        topics, probs = self.model.fit_transform(texts)
        return topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        return self.model.get_topic_info()


# ── UNIFIED PIPELINE ──────────────────────────────────────────────────────────

# Business-relevant topic label mapping for auto-labeling
BUSINESS_TOPIC_LABELS = [
    "Product Quality",
    "Shipping & Delivery",
    "Customer Service",
    "Value for Money",
    "Ease of Use",
    "Product Defects",
    "Product Appearance",
    "Overall Experience",
]


def run_topic_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    compound_col: str = "compound",
    n_topics: int = 8,
    backend: str = "lda",
) -> tuple[pd.DataFrame, TopicModelResult]:
    """
    Run topic modeling on review text.
    Returns (enriched_df, TopicModelResult)
    """
    texts = df[text_col].tolist()
    print(f"🔍 Running {backend} topic modeling on {len(texts)} documents...")

    if backend == "bertopic":
        bert = BERTopicModel()
        if bert.available:
            return _run_bertopic(df, texts, bert, compound_col)
        else:
            print("↩️  Falling back to LDA")

    return _run_lda(df, texts, compound_col, n_topics)


def _run_lda(
    df: pd.DataFrame,
    texts: list[str],
    compound_col: str,
    n_topics: int,
) -> tuple[pd.DataFrame, TopicModelResult]:
    lda = LDATopicModel(n_topics=n_topics)
    lda.fit(texts)

    doc_topic_matrix = lda.transform(texts)
    doc_topics = doc_topic_matrix.argmax(axis=1).tolist()
    doc_probs  = doc_topic_matrix.max(axis=1).tolist()

    all_keywords = lda.get_topic_keywords()
    compounds    = df[compound_col].tolist() if compound_col in df else [0.0] * len(texts)

    topics = []
    for i, keywords in enumerate(all_keywords):
        mask = [j for j, t in enumerate(doc_topics) if t == i]
        size = len(mask)
        sentiment_avg = float(np.mean([compounds[j] for j in mask])) if mask else 0.0
        label = lda.auto_label(keywords)

        # Coherence approximation: avg top-word weight
        coherence = float(np.mean([w for _, w in keywords[:5]]))

        topics.append(Topic(
            id=i, label=label, keywords=keywords,
            size=size, sentiment_avg=sentiment_avg, coherence=coherence,
        ))

    coverage = sum(1 for p in doc_probs if p > 0.3) / len(doc_probs)

    result = TopicModelResult(
        topics=topics, doc_topics=doc_topics, doc_probs=doc_probs,
        model_type="lda", n_topics=n_topics, coverage=coverage,
    )

    df = df.copy()
    df["topic_id"]    = doc_topics
    df["topic_prob"]  = doc_probs
    df["topic_label"] = [topics[t].label for t in doc_topics]

    print(f"✅ LDA complete: {n_topics} topics, {coverage:.1%} coverage")
    return df, result


def _run_bertopic(
    df: pd.DataFrame,
    texts: list[str],
    bert: BERTopicModel,
    compound_col: str,
) -> tuple[pd.DataFrame, TopicModelResult]:
    doc_topics, probs = bert.fit_transform(texts)
    topic_info = bert.get_topic_info()
    compounds  = df[compound_col].tolist() if compound_col in df else [0.0] * len(texts)

    topics = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue  # noise cluster
        mask = [j for j, t in enumerate(doc_topics) if t == tid]
        keywords = [(w, s) for w, s in bert.model.get_topic(tid)][:10]
        sentiment_avg = float(np.mean([compounds[j] for j in mask])) if mask else 0.0

        topics.append(Topic(
            id=tid,
            label=row.get("Name", f"Topic {tid}"),
            keywords=keywords,
            size=row["Count"],
            sentiment_avg=sentiment_avg,
            coherence=0.0,
        ))

    doc_topic_labels = []
    label_map = {t.id: t.label for t in topics}
    for tid in doc_topics:
        doc_topic_labels.append(label_map.get(tid, "Other"))

    noise_count = sum(1 for t in doc_topics if t == -1)
    coverage    = 1 - noise_count / len(doc_topics)

    result = TopicModelResult(
        topics=topics, doc_topics=doc_topics,
        doc_probs=[float(p) if p else 0.0 for p in probs],
        model_type="bertopic", n_topics=len(topics), coverage=coverage,
    )

    df = df.copy()
    df["topic_id"]    = doc_topics
    df["topic_prob"]  = result.doc_probs
    df["topic_label"] = doc_topic_labels

    print(f"✅ BERTopic complete: {len(topics)} topics, {coverage:.1%} coverage")
    return df, result
