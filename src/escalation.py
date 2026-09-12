"""
Escalation logic: interpretable rule/threshold system.

Combines multiple signals to decide auto-handle vs. escalate-to-human.
Every decision includes a plain-English reason string.

This is deliberately NOT a black-box model — an interviewer should be
able to trace exactly why any given message was escalated or auto-handled.

Signals used:
  1. Intent risk tier (high/medium/low)
  2. Classifier confidence (0–1)
  3. Retrieval match strength (top-1 cosine similarity)
  4. Trigger keywords / sentiment
  5. Message complexity (length, question marks, multi-issue indicators)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Thresholds (tunable) ─────────────────────────────────────────────────

# Classifier confidence below this → escalate
CONFIDENCE_THRESHOLD = 0.25  # maps to LLM confidence ≤ 2/5

# Retrieval similarity below this → novel issue → escalate
SIMILARITY_THRESHOLD = 0.35

# ── Trigger patterns ─────────────────────────────────────────────────────

# High-priority keywords that suggest the message needs human attention
_ESCALATION_KEYWORDS = re.compile(
    r"(?i)\b("
    r"cancel|refund|charged|lawsuit|attorney|lawyer|legal|"
    r"fraud|stolen|hacked|unauthorized|"
    r"discrimination|harass|threat|police|"
    r"disability|disabled|accessib|"
    r"dead|died|death|emergency|urgent|"
    r"manager|supervisor|escalat"
    r")\b"
)

# Strong negative sentiment
_STRONG_NEGATIVE = re.compile(
    r"(?i)\b("
    r"worst|terrible|horrible|disgusting|pathetic|useless|incompetent|"
    r"scam|rip\s*off|robbery|"
    r"never\s+again|done\s+with|switching\s+to"
    r")\b"
)

# Multi-issue indicators
_MULTI_ISSUE = re.compile(
    r"(?i)\b(also|another\s+issue|on\s+top\s+of|additionally|plus|and\s+also)\b"
)


def decide_escalation(
    message: str,
    intent_key: str,
    confidence: float,
    top_similarity: float,
    risk_tier: str = "low",
) -> tuple[str, str]:
    """
    Decide whether to auto-handle or escalate a customer message.

    Args:
        message: The customer's message text.
        intent_key: The classified intent key.
        confidence: Classifier confidence score (0.0–1.0).
        top_similarity: Similarity score of the best retrieval match (0.0–1.0).
        risk_tier: Intent risk tier from taxonomy ("high"/"medium"/"low").

    Returns:
        Tuple of (decision, reason) where:
            - decision: "auto_handle" or "escalate"
            - reason: Plain-English explanation of why.
    """
    reasons_to_escalate = []

    # ── Signal 1: Intent risk tier ───────────────────────────────────
    if risk_tier == "high" and confidence < 0.75:
        reasons_to_escalate.append(
            f"High-risk intent ({intent_key}) with moderate confidence ({confidence:.0%})"
        )

    # ── Signal 2: Classifier confidence ──────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        reasons_to_escalate.append(
            f"Very low classifier confidence ({confidence:.0%}) — ambiguous intent"
        )

    # ── Signal 3: Retrieval match strength ───────────────────────────
    if top_similarity < SIMILARITY_THRESHOLD:
        reasons_to_escalate.append(
            f"No strong precedent match (best similarity: {top_similarity:.2f}) — "
            f"may be a novel issue"
        )

    # ── Signal 4: Trigger keywords ───────────────────────────────────
    kw_match = _ESCALATION_KEYWORDS.search(message)
    if kw_match:
        reasons_to_escalate.append(
            f"Contains escalation keyword: '{kw_match.group()}'"
        )

    neg_match = _STRONG_NEGATIVE.search(message)
    if neg_match:
        reasons_to_escalate.append(
            f"Strong negative sentiment detected: '{neg_match.group()}'"
        )

    # ── Signal 5: Message complexity ─────────────────────────────────
    multi_match = _MULTI_ISSUE.search(message)
    if multi_match:
        reasons_to_escalate.append(
            "Message mentions multiple issues — may need comprehensive triage"
        )

    # Very short messages are often ambiguous
    if len(message.strip()) < 15:
        reasons_to_escalate.append(
            "Very short message — insufficient context for automated handling"
        )

    # ── Decision ─────────────────────────────────────────────────────
    if reasons_to_escalate:
        reason = "ESCALATE because: " + "; ".join(reasons_to_escalate) + "."
        return "escalate", reason
    else:
        # Build a positive reason for auto-handling
        reason = (
            f"AUTO-HANDLE: Intent '{intent_key}' classified with "
            f"{confidence:.0%} confidence (risk tier: {risk_tier}), "
            f"strong precedent match ({top_similarity:.2f} similarity), "
            f"no escalation triggers detected."
        )
        return "auto_handle", reason


def get_escalation_summary(decision: str, reason: str) -> str:
    """Format a concise summary of the escalation decision."""
    icon = "🔴" if decision == "escalate" else "🟢"
    return f"{icon} {decision.upper()}: {reason}"
