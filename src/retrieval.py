"""
Retrieval module: embedding index + top-k retrieval over resolved precedents.

Embeds all resolved customer→company pairs using sentence-transformers
and retrieves the most similar precedents for a new customer message.

All retrieved precedents are PII-scrubbed before being returned.

Design choices:
  - all-MiniLM-L6-v2: fast, local, no API cost (DECISION_LOG.md #2)
  - numpy cosine similarity: FAISS is overkill at this scale (#3)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.pii_scrubber import scrub_precedent

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_CACHE = DATA_DIR / "embeddings.npy"


class RetrievalIndex:
    """
    Embedding-based retrieval index over resolved customer→company pairs.

    On initialization, loads resolved pairs and computes embeddings
    (cached to disk for fast restarts). At query time, finds the top-k
    most similar customer messages and returns PII-scrubbed precedents.
    """

    def __init__(
        self,
        resolved_pairs_path: str | Path = DATA_DIR / "resolved_pairs.csv",
        model_name: str = MODEL_NAME,
        force_recompute: bool = False,
    ):
        self.pairs_df = pd.read_csv(resolved_pairs_path)
        logger.info("Loaded %d resolved pairs for retrieval", len(self.pairs_df))

        # Load or compute embeddings
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_compute_embeddings(force_recompute)
        logger.info("Embedding index ready: %s", self.embeddings.shape)

    def _load_or_compute_embeddings(self, force: bool) -> np.ndarray:
        """Load cached embeddings or compute fresh ones."""
        if not force and EMBEDDINGS_CACHE.exists():
            logger.info("Loading cached embeddings from %s", EMBEDDINGS_CACHE)
            return np.load(EMBEDDINGS_CACHE)

        logger.info("Computing embeddings for %d customer messages...",
                     len(self.pairs_df))
        texts = self.pairs_df["customer_text"].fillna("").tolist()
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,  # pre-normalize for cosine similarity
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Cache to disk
        np.save(EMBEDDINGS_CACHE, embeddings)
        logger.info("Saved embeddings to %s", EMBEDDINGS_CACHE)
        return embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve the top-k most similar resolved precedents for a query.

        Args:
            query: The new customer message to find precedents for.
            top_k: Number of precedents to return.

        Returns:
            List of dicts, each containing:
                - customer_text: PII-scrubbed customer message from precedent
                - company_text: PII-scrubbed company reply from precedent
                - similarity: cosine similarity score (0 to 1)
        """
        # Embed the query
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )[0]

        # Cosine similarity (embeddings are pre-normalized, so dot product = cosine)
        similarities = self.embeddings @ query_embedding

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.pairs_df.iloc[idx]
            # PII-scrub the precedent before returning
            scrubbed_customer, scrubbed_company = scrub_precedent(
                str(row["customer_text"]),
                str(row["company_text"]),
            )
            results.append({
                "customer_text": scrubbed_customer,
                "company_text": scrubbed_company,
                "similarity": float(similarities[idx]),
                "original_index": int(idx),
            })

        return results

    def get_top_similarity(self, query: str) -> float:
        """Get the similarity score of the best match (for escalation logic)."""
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )[0]
        similarities = self.embeddings @ query_embedding
        return float(np.max(similarities))


# ── Module-level singleton (lazy init) ───────────────────────────────────

_index: RetrievalIndex | None = None


def get_index(force_reload: bool = False) -> RetrievalIndex:
    """Get or create the module-level retrieval index singleton."""
    global _index
    if _index is None or force_reload:
        _index = RetrievalIndex()
    return _index


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Convenience function: retrieve top-k precedents for a query."""
    return get_index().retrieve(query, top_k)
