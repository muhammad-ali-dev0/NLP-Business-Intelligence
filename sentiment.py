"""
nlp/sentiment.py
----------------
Sentiment analysis module supporting two backends:
  1. Transformer-based (DistilBERT) — production quality
  2. VADER lexicon — fast, no GPU required (fallback/demo)

Both return a unified SentimentResult dataclass.
"""

from __future__ import annotations
import re
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── DATA MODEL ────────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label:      Literal["positive", "neutral", "negative"]
    score:      float       # confidence in predicted label (0–1)
    positive:   float       # raw positive probability
    neutral:    float       # raw neutral probability
    negative:   float       # raw negative probability
    compound:   float       # unified score −1 (negative) → +1 (positive)
    aspects:    dict = field(default_factory=dict)  # aspect → sentiment


# ── VADER BACKEND (fast, no download) ─────────────────────────────────────────

class VADERAnalyzer:
    """
    Lexicon-based sentiment using VADER.
    Suitable for social media / review text without fine-tuning.
    Speed: ~50K reviews/sec on CPU.
    """

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.available = True
        except ImportError:
            self.available = False
            self.analyzer = None

    def analyze(self, text: str) -> SentimentResult:
        if not self.available or self.analyzer is None:
            return self._rule_based(text)

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label, score = "positive", scores["pos"]
        elif compound <= -0.05:
            label, score = "negative", scores["neg"]
        else:
            label, score = "neutral", scores["neu"]

        return SentimentResult(
            label=label,
            score=max(scores["pos"], scores["neu"], scores["neg"]),
            positive=scores["pos"],
            neutral=scores["neu"],
            negative=scores["neg"],
            compound=compound,
        )

    def _rule_based(self, text: str) -> SentimentResult:
        """Pure rule-based fallback when VADER not installed."""
        text_lower = text.lower()
        pos_words = ["great", "excellent", "love", "perfect", "amazing", "good",
                     "fantastic", "outstanding", "best", "wonderful", "highly"]
        neg_words = ["terrible", "awful", "hate", "broken", "worst", "bad",
                     "disappointed", "poor", "waste", "horrible", "useless"]

        pos = sum(1 for w in pos_words if w in text_lower)
        neg = sum(1 for w in neg_words if w in text_lower)
        total = pos + neg + 1e-9

        compound = (pos - neg) / (pos + neg + 1e-9)
        compound = max(-1.0, min(1.0, compound))

        if compound > 0.1:
            label = "positive"
        elif compound < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            label=label,
            score=abs(compound),
            positive=pos / total,
            neutral=1 / total,
            negative=neg / total,
            compound=compound,
        )


# ── TRANSFORMER BACKEND ───────────────────────────────────────────────────────

class TransformerAnalyzer:
    """
    DistilBERT fine-tuned on SST-2 for sentiment classification.
    Model: distilbert-base-uncased-finetuned-sst-2-english
    Speed: ~500 reviews/sec on CPU, ~5K on GPU.
    Accuracy: 91.3% on SST-2 benchmark.

    Falls back to VADER if transformers not installed.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, device: str = "auto"):
        self.pipe = None
        self.vader = VADERAnalyzer()
        try:
            from transformers import pipeline
            import torch
            dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
            self.pipe = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                device=0 if dev == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            print(f"✅ TransformerAnalyzer loaded on {dev}")
        except Exception as e:
            print(f"⚠️  Transformer not available ({e}), falling back to VADER.")

    def analyze(self, text: str) -> SentimentResult:
        if self.pipe is None:
            return self.vader.analyze(text)

        result = self.pipe(text[:512])[0]
        label_raw = result["label"].lower()  # "POSITIVE" or "NEGATIVE"
        confidence = result["score"]

        # Map binary labels → ternary
        if label_raw == "positive" and confidence > 0.80:
            label = "positive"
            compound = confidence
        elif label_raw == "negative" and confidence > 0.80:
            label = "negative"
            compound = -confidence
        else:
            label = "neutral"
            compound = 0.0

        pos = confidence if label_raw == "positive" else 1 - confidence
        neg = confidence if label_raw == "negative" else 1 - confidence
        neu = 1 - abs(compound)

        return SentimentResult(
            label=label,
            score=confidence,
            positive=round(pos, 4),
            neutral=round(max(neu, 0), 4),
            negative=round(neg, 4),
            compound=round(compound, 4),
        )

    def analyze_batch(self, texts: list[str], batch_size: int = 64) -> list[SentimentResult]:
        if self.pipe is None:
            return [self.vader.analyze(t) for t in texts]

        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:512] for t in texts[i:i + batch_size]]
            raw = self.pipe(batch)
            for r in raw:
                label_raw = r["label"].lower()
                conf = r["score"]
                compound = conf if label_raw == "positive" else -conf
                if abs(compound) < 0.80:
                    label = "neutral"
                else:
                    label = label_raw
                results.append(SentimentResult(
                    label=label, score=conf,
                    positive=conf if label_raw == "positive" else 1 - conf,
                    neutral=max(1 - abs(compound), 0),
                    negative=conf if label_raw == "negative" else 1 - conf,
                    compound=round(compound, 4),
                ))
        return results


# ── ASPECT-BASED SENTIMENT ─────────────────────────────────────────────────────

ASPECT_KEYWORDS = {
    "quality":        ["quality", "build", "durable", "material", "solid", "cheap", "flimsy"],
    "shipping":       ["shipping", "delivery", "arrived", "package", "fast", "late", "transit"],
    "customer_service": ["service", "support", "helpful", "return", "refund", "response"],
    "value":          ["price", "value", "worth", "expensive", "affordable", "money", "cost"],
    "ease_of_use":    ["easy", "simple", "intuitive", "difficult", "instructions", "setup"],
}


def extract_aspect_sentiments(text: str, analyzer: VADERAnalyzer) -> dict[str, float]:
    """
    Extract per-aspect sentiment scores by windowing around aspect keywords.
    Returns dict of {aspect: compound_score}
    """
    sentences = re.split(r"[.!?]+", text.lower())
    aspect_scores: dict[str, list[float]] = {a: [] for a in ASPECT_KEYWORDS}

    for sentence in sentences:
        if not sentence.strip():
            continue
        result = analyzer.analyze(sentence)
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in sentence for kw in keywords):
                aspect_scores[aspect].append(result.compound)

    return {
        aspect: float(np.mean(scores)) if scores else 0.0
        for aspect, scores in aspect_scores.items()
    }


# ── BATCH PIPELINE ────────────────────────────────────────────────────────────

def run_sentiment_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    backend: Literal["vader", "transformer"] = "vader",
) -> pd.DataFrame:
    """
    Run sentiment analysis on a DataFrame of reviews.
    Adds columns: sentiment_label, sentiment_score, compound,
                  aspect_quality, aspect_shipping, aspect_service,
                  aspect_value, aspect_ease_of_use
    """
    if backend == "transformer":
        analyzer_obj = TransformerAnalyzer()
    else:
        analyzer_obj = VADERAnalyzer()

    vader = VADERAnalyzer()  # always needed for aspect extraction

    texts = df[text_col].tolist()
    print(f"🔍 Running {backend} sentiment on {len(texts)} reviews...")

    if backend == "transformer" and hasattr(analyzer_obj, "analyze_batch"):
        results = analyzer_obj.analyze_batch(texts)
    else:
        results = [analyzer_obj.analyze(t) for t in texts]

    # Aspect sentiment (always VADER for speed)
    aspects = [extract_aspect_sentiments(t, vader) for t in texts]

    df = df.copy()
    df["sentiment_label"]    = [r.label    for r in results]
    df["sentiment_score"]    = [r.score    for r in results]
    df["compound"]           = [r.compound for r in results]
    df["prob_positive"]      = [r.positive for r in results]
    df["prob_neutral"]       = [r.neutral  for r in results]
    df["prob_negative"]      = [r.negative for r in results]

    for aspect in ASPECT_KEYWORDS:
        df[f"aspect_{aspect}"] = [a[aspect] for a in aspects]

    print(f"✅ Sentiment complete. Distribution: "
          f"{df['sentiment_label'].value_counts().to_dict()}")
    return df
