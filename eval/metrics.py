"""
Evaluation metrics: classification, groundedness proxy, and escalation.

Computes:
  - Intent classification: accuracy, macro-F1, per-class precision/recall,
    confusion matrix
  - Groundedness proxy: ROUGE-L overlap between draft reply and top precedent
  - Escalation: accuracy, precision/recall/F1 for "escalate" class
    (asymmetric cost: false auto-handle >> false escalate)
"""

import logging
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute intent classification metrics.

    Returns dict with: accuracy, macro_f1, per_class_report, confusion_matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class report
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "class_distribution": dict(Counter(y_true)),
    }


def escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute escalation decision metrics.

    Emphasizes precision/recall for the "escalate" class since
    a false auto-handle (missing a message that should have been
    escalated) is much costlier than a false escalate.
    """
    # Binary: "escalate" = positive class
    accuracy = accuracy_score(y_true, y_pred)

    # Escalate-class metrics
    escalate_precision = precision_score(
        y_true, y_pred, pos_label="escalate", zero_division=0
    )
    escalate_recall = recall_score(
        y_true, y_pred, pos_label="escalate", zero_division=0
    )
    escalate_f1 = f1_score(
        y_true, y_pred, pos_label="escalate", zero_division=0
    )

    # Confusion matrix
    labels = ["auto_handle", "escalate"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Extract specific error types
    n_false_auto = 0  # should have been escalated but wasn't
    n_false_escalate = 0  # didn't need escalation but got it
    for true, pred in zip(y_true, y_pred):
        if true == "escalate" and pred == "auto_handle":
            n_false_auto += 1
        elif true == "auto_handle" and pred == "escalate":
            n_false_escalate += 1

    return {
        "accuracy": round(accuracy, 4),
        "escalate_precision": round(escalate_precision, 4),
        "escalate_recall": round(escalate_recall, 4),
        "escalate_f1": round(escalate_f1, 4),
        "false_auto_handle": n_false_auto,  # the dangerous errors
        "false_escalate": n_false_escalate,  # the costly-but-safe errors
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "class_distribution": dict(Counter(y_true)),
    }


def groundedness_score(reply: str, precedent_text: str) -> float:
    """
    Compute a groundedness proxy: ROUGE-L between the draft reply
    and the top retrieved precedent.

    This is a rough proxy — high overlap suggests the reply draws
    from the precedent; low overlap might mean the reply is ungrounded
    OR that the LLM paraphrased well. It's reported with this caveat.

    Returns a score between 0 and 1.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(precedent_text, reply)
        return round(scores["rougeL"].fmeasure, 4)
    except ImportError:
        logger.warning("rouge_score not installed; using token overlap fallback")
        return _token_overlap(reply, precedent_text)


def _token_overlap(text_a: str, text_b: str) -> float:
    """Fallback: simple token-level Jaccard overlap."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return round(len(intersection) / len(union), 4)


def batch_groundedness(
    replies: list[str],
    precedent_texts: list[str],
) -> dict:
    """Compute groundedness scores for a batch of replies."""
    scores = [
        groundedness_score(r, p) for r, p in zip(replies, precedent_texts)
    ]
    return {
        "mean": round(np.mean(scores), 4),
        "median": round(np.median(scores), 4),
        "std": round(np.std(scores), 4),
        "min": round(np.min(scores), 4),
        "max": round(np.max(scores), 4),
        "scores": scores,
    }
