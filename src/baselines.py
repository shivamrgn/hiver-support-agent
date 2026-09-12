"""
Baselines for intent classification and reply quality.

Provides trivial and simple baselines to compare against the main system:
  - Intent: majority-class (trivial) + TF-IDF logistic regression (simple)
  - Reply: canned reply (trivial) + ungrounded zero-shot LLM (simple)
  - Escalation: always-escalate, always-auto-handle, keyword-only

The zero-shot LLM baseline doubles as a clean ablation showing
what retrieval grounding actually buys.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline

from src.llm_client import complete, cached_complete
from src.intent_classifier import INTENT_TAXONOMY

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# INTENT BASELINES
# ══════════════════════════════════════════════════════════════════════════

class MajorityClassBaseline:
    """
    Trivial baseline: always predict the most common intent.

    This is the floor — any useful classifier must beat this.
    """

    def __init__(self, majority_class: str = "other"):
        self.majority_class = majority_class

    def fit(self, labels: list[str]):
        """Set majority class from label distribution."""
        counts = Counter(labels)
        self.majority_class = counts.most_common(1)[0][0]
        logger.info("Majority class baseline: '%s' (%.1f%%)",
                     self.majority_class,
                     counts[self.majority_class] / len(labels) * 100)
        return self

    def predict(self, messages: list[str]) -> list[str]:
        """Always return the majority class."""
        return [self.majority_class] * len(messages)

    def predict_one(self, message: str) -> str:
        return self.majority_class


class TfidfLogregBaseline:
    """
    Simple baseline: TF-IDF vectorization + Logistic Regression.

    Trained on the few-shot examples from the intent taxonomy plus
    any additional labeled data available.
    """

    def __init__(self):
        self.pipeline = SkPipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self.is_fitted = False

    def fit(self, texts: list[str], labels: list[str]):
        """Fit the TF-IDF + LogReg pipeline."""
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        logger.info("TF-IDF + LogReg fitted on %d examples", len(texts))
        return self

    def fit_from_taxonomy(self):
        """
        Bootstrap training data from the taxonomy examples.

        This creates a small but representative training set from the
        2-3 examples per category in INTENT_TAXONOMY.
        """
        texts, labels = [], []
        for key, info in INTENT_TAXONOMY.items():
            for example in info["examples"]:
                texts.append(example)
                labels.append(key)
        return self.fit(texts, labels)

    def predict(self, messages: list[str]) -> list[str]:
        """Predict intents for a batch of messages."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(messages).tolist()

    def predict_one(self, message: str) -> str:
        return self.predict([message])[0]


# ══════════════════════════════════════════════════════════════════════════
# REPLY BASELINES
# ══════════════════════════════════════════════════════════════════════════

# Trivial baseline: one fixed canned reply for everything
CANNED_REPLY = (
    "Thanks for reaching out! We're sorry to hear you're having trouble. "
    "Please send us a DM with your account details so we can look into "
    "this for you. We're here to help! ^SK"
)


def canned_reply_baseline(message: str) -> str:
    """Trivial baseline: return the same canned reply regardless of input."""
    return CANNED_REPLY


def zero_shot_llm_baseline(
    message: str,
    use_cache: bool = True,
) -> str:
    """
    Simple baseline: zero-shot LLM reply with NO retrieval grounding.

    This is the clean ablation — comparing this to the grounded reply
    shows what retrieval actually buys.
    """
    system_prompt = """You are a helpful customer support agent for a music streaming service.
Reply to the customer message below. Be professional, empathetic, and helpful.
Keep your response concise (under 280 characters if possible, like a tweet)."""

    user_prompt = f'Customer message: "{message}"\n\nDraft a helpful reply:'

    cache_key = f"baseline_reply_{hash(message)}"

    if use_cache:
        response = cached_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=200,
            cache_key=cache_key,
            cache_file="baseline_reply_cache.json",
        )
    else:
        response = complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=200,
        )

    return response.strip().strip('"').strip("'")


# ══════════════════════════════════════════════════════════════════════════
# ESCALATION BASELINES
# ══════════════════════════════════════════════════════════════════════════

def always_escalate(message: str) -> tuple[str, str]:
    """Trivial baseline: escalate everything."""
    return "escalate", "Baseline: always escalate"


def always_auto_handle(message: str) -> tuple[str, str]:
    """Trivial baseline: auto-handle everything."""
    return "auto_handle", "Baseline: always auto-handle"


# Simple escalation keywords (subset of the full escalation module)
_SIMPLE_KEYWORDS = re.compile(
    r"(?i)\b(cancel|refund|fraud|hacked|stolen|lawsuit|lawyer|manager)\b"
)


def keyword_only_escalation(message: str) -> tuple[str, str]:
    """Simple baseline: escalate only if trigger keywords are present."""
    match = _SIMPLE_KEYWORDS.search(message)
    if match:
        return "escalate", f"Keyword match: '{match.group()}'"
    return "auto_handle", "No escalation keywords found"
