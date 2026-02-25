"""
nlp/embeddings.py
-----------------
Sentence embedding + dimensionality reduction for visualization.

Pipeline:
  text → SentenceTransformer (all-MiniLM-L6-v2) → 384-dim embeddings
       → UMAP → 2D scatter plot colored by sentiment / topic

Falls back to TF-IDF + PCA if sentence-transformers not installed.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── EMBEDDING BACKENDS ────────────────────────────────────────────────────────

class SentenceEmbedder:
    """
    Generates 384-dim sentence embeddings using all-MiniLM-L6-v2.
    Semantic similarity preserved in embedding space.
    Processes ~2,000 reviews/sec on CPU.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model     = None
        self.available = False
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model     = SentenceTransformer(self.MODEL_NAME)
            self.available = True
            print(f"✅ SentenceTransformer loaded: {self.MODEL_NAME}")
        except ImportError:
            print("⚠️  sentence-transformers not installed. Using TF-IDF fallback.")

    def encode(self, texts: list[str], batch_size: int = 128,
               show_progress: bool = True) -> np.ndarray:
        if self.available and self.model:
            return self.model.encode(
                texts, batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )
        return self._tfidf_fallback(texts)

    def _tfidf_fallback(self, texts: list[str]) -> np.ndarray:
        """TF-IDF + SVD as embedding fallback. Lower quality but dependency-free."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        tfidf = vec.fit_transform(texts)
        svd   = TruncatedSVD(n_components=min(100, tfidf.shape[1] - 1), random_state=42)
        reduced = svd.fit_transform(tfidf)
        return normalize(reduced).astype(np.float32)


# ── DIMENSIONALITY REDUCTION ──────────────────────────────────────────────────

class DimensionReducer:
    """
    Reduces high-dimensional embeddings to 2D for visualization.
    Primary: UMAP (preserves local + global structure)
    Fallback: t-SNE (slower, good local structure)
    Fast fallback: PCA (linear, fast)
    """

    def __init__(self, method: str = "umap", n_neighbors: int = 15,
                 min_dist: float = 0.1, random_state: int = 42):
        self.method       = method
        self.n_neighbors  = n_neighbors
        self.min_dist     = min_dist
        self.random_state = random_state
        self.reducer      = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns (n, 2) array of 2D coordinates."""
        if self.method == "umap":
            return self._umap(embeddings)
        elif self.method == "tsne":
            return self._tsne(embeddings)
        else:
            return self._pca(embeddings)

    def _umap(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            from umap import UMAP
            reducer = UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=2,
                metric="cosine",
                random_state=self.random_state,
                verbose=False,
            )
            print("🔍 Running UMAP dimensionality reduction...")
            result = reducer.fit_transform(embeddings)
            print("✅ UMAP complete")
            return result
        except ImportError:
            print("⚠️  UMAP not installed, falling back to t-SNE")
            return self._tsne(embeddings)

    def _tsne(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        # PCA first to speed up t-SNE
        from sklearn.decomposition import PCA
        if embeddings.shape[1] > 50:
            pca  = PCA(n_components=50, random_state=self.random_state)
            embeddings = pca.fit_transform(embeddings)

        print("🔍 Running t-SNE (this may take a moment)...")
        tsne = TSNE(
            n_components=2, perplexity=30,
            n_iter=1000, random_state=self.random_state,
            verbose=0,
        )
        result = tsne.fit_transform(embeddings)
        print("✅ t-SNE complete")
        return result

    def _pca(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        return pca.fit_transform(embeddings)


# ── SIMILARITY SEARCH ─────────────────────────────────────────────────────────

def find_similar_reviews(
    query: str,
    embedder: SentenceEmbedder,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 5,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Semantic search: find reviews most similar to a query string.
    Uses cosine similarity in embedding space.
    """
    query_emb = embedder.encode([query], show_progress=False)[0]

    # Cosine similarity (embeddings are L2-normalized)
    sims = embeddings @ query_emb
    top_indices = sims.argsort()[-top_k:][::-1]

    result = df.iloc[top_indices].copy()
    result["similarity"] = sims[top_indices].round(4)
    return result[[text_col, "sentiment_label", "stars", "category", "similarity"]]


# ── EMBEDDING STATISTICS ──────────────────────────────────────────────────────

def compute_cluster_stats(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
) -> pd.DataFrame:
    """
    Compute centroid and spread for each topic/sentiment cluster.
    Used for annotation on the scatter plot.
    """
    df = df.copy()
    df["umap_x"] = coords_2d[:, 0]
    df["umap_y"] = coords_2d[:, 1]

    if "topic_label" not in df.columns:
        return df

    stats = df.groupby("topic_label").agg(
        centroid_x=("umap_x", "mean"),
        centroid_y=("umap_y", "mean"),
        count=("umap_x", "count"),
        avg_sentiment=("compound", "mean") if "compound" in df else ("umap_x", "count"),
    ).reset_index()

    return df, stats


# ── FULL EMBEDDING PIPELINE ───────────────────────────────────────────────────

def run_embedding_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    reduction_method: str = "umap",
    sample_n: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full pipeline: text → embeddings → 2D coords.
    Returns (df with umap_x/umap_y, raw_embeddings)
    """
    from typing import Optional

    # Optionally sample for speed in development
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"📊 Sampled {sample_n} reviews for embedding visualization")

    texts = df[text_col].tolist()

    # Step 1: Embed
    embedder  = SentenceEmbedder()
    embeddings = embedder.encode(texts)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Step 2: Reduce to 2D
    reducer = DimensionReducer(method=reduction_method)
    coords  = reducer.fit_transform(embeddings)

    df = df.copy()
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    return df, embeddings


# Fix missing Optional import at module level
from typing import Optional
