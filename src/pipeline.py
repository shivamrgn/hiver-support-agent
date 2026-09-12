"""
End-to-end pipeline: message → {intent, reply_draft, escalation, reason}.

This is the main orchestrator that wires together:
  intent_classifier → retrieval → reply_generator → escalation

Usage:
    from src.pipeline import end_to_end
    result = end_to_end("I can't log into my account!")
"""

import logging
from src.intent_classifier import classify_intent, get_intent_label, get_risk_tier
from src.retrieval import get_index
from src.reply_generator import generate_reply
from src.escalation import decide_escalation

logger = logging.getLogger(__name__)


def end_to_end(
    message: str,
    use_cache: bool = True,
    top_k: int = 3,
) -> dict:
    """
    Process a customer message through the full agent pipeline.

    Steps:
        1. Classify intent (LLM few-shot)
        2. Retrieve similar resolved precedents (embedding similarity)
        3. Generate reply draft (LLM, grounded in precedents)
        4. Decide auto-handle vs. escalate (rule-based)

    Args:
        message: The incoming customer message.
        use_cache: If True, use cached LLM responses (fast path).
                   If False, call LLM APIs fresh (--fresh flag).
        top_k: Number of precedents to retrieve.

    Returns:
        Dict with keys:
            - intent: intent key (e.g., "account_access")
            - intent_label: human-readable intent label
            - confidence: classifier confidence (0.0–1.0)
            - reply_draft: generated reply text
            - escalation_decision: "auto_handle" or "escalate"
            - escalation_reason: plain-English reason string
            - retrieved_precedents: list of PII-scrubbed precedent dicts
            - retrieval_scores: list of similarity scores
    """
    # Step 1: Classify intent
    intent_key, confidence = classify_intent(message, use_cache=use_cache)
    intent_label = get_intent_label(intent_key)
    risk_tier = get_risk_tier(intent_key)

    logger.info("Intent: %s (%.0f%% confidence, risk: %s)",
                intent_label, confidence * 100, risk_tier)

    # Step 2: Retrieve precedents
    index = get_index()
    precedents = index.retrieve(message, top_k=top_k)
    top_similarity = precedents[0]["similarity"] if precedents else 0.0
    retrieval_scores = [p["similarity"] for p in precedents]

    logger.info("Top retrieval similarity: %.3f", top_similarity)

    # Step 3: Generate reply draft
    reply_draft = generate_reply(
        customer_message=message,
        intent_label=intent_label,
        precedents=precedents,
        use_cache=use_cache,
    )

    # Step 4: Escalation decision
    escalation_decision, escalation_reason = decide_escalation(
        message=message,
        intent_key=intent_key,
        confidence=confidence,
        top_similarity=top_similarity,
        risk_tier=risk_tier,
    )

    logger.info("Escalation: %s", escalation_decision)

    return {
        "intent": intent_key,
        "intent_label": intent_label,
        "confidence": confidence,
        "reply_draft": reply_draft,
        "escalation_decision": escalation_decision,
        "escalation_reason": escalation_reason,
        "retrieved_precedents": precedents,
        "retrieval_scores": retrieval_scores,
    }


def batch_process(
    messages: list[str],
    use_cache: bool = True,
    top_k: int = 3,
) -> list[dict]:
    """
    Process a batch of customer messages through the pipeline.

    Args:
        messages: List of customer message strings.
        use_cache: Whether to use cached LLM responses.
        top_k: Number of precedents to retrieve per message.

    Returns:
        List of result dicts (same structure as end_to_end output).
    """
    from tqdm import tqdm

    results = []
    for msg in tqdm(messages, desc="Processing messages"):
        try:
            result = end_to_end(msg, use_cache=use_cache, top_k=top_k)
        except Exception as e:
            logger.error("Failed to process message: %s — Error: %s",
                         msg[:50], e)
            result = {
                "intent": "other",
                "intent_label": "Other / General Inquiry",
                "confidence": 0.0,
                "reply_draft": "",
                "escalation_decision": "escalate",
                "escalation_reason": f"Pipeline error: {e}",
                "retrieved_precedents": [],
                "retrieval_scores": [],
            }
        results.append(result)
    return results
