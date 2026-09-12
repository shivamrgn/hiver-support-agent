"""
Intent classifier using LLM few-shot classification.

Classifies incoming customer messages into a brand-specific intent taxonomy.
Returns (intent_label, confidence_score).

Design choice: LLM few-shot over fine-tuned classifier because we don't have
a pre-labeled training set, and few-shot avoids the cold-start problem.
TF-IDF + logistic regression serves as the simple baseline.
See DECISION_LOG.md #5.
"""

import json
import logging
import re
from pathlib import Path

from src.llm_client import complete, cached_complete

logger = logging.getLogger(__name__)

# ── Intent Taxonomy ─────────────────────────────────────────────────────
# This will be populated after Phase 2 analysis. The structure below is
# the template that gets filled in from actual data inspection.

INTENT_TAXONOMY = {
    "account_access": {
        "label": "Account Access / Login Issues",
        "definition": "Customer cannot log in, account locked, password reset, account hacked or compromised.",
        "examples": [
            "I can't log into my account, keeps saying wrong password",
            "My account got hacked and someone changed my email",
            "I've been locked out of my account for 3 days now",
        ],
        "risk_tier": "high",  # for escalation logic
    },
    "subscription_billing": {
        "label": "Subscription & Billing",
        "definition": "Issues with payments, charges, plan changes, refunds, free trial problems.",
        "examples": [
            "I was charged twice this month for my premium subscription",
            "How do I cancel my subscription?",
            "I signed up for a free trial but got charged immediately",
        ],
        "risk_tier": "high",
    },
    "playback_streaming": {
        "label": "Playback / Streaming Issues",
        "definition": "Songs won't play, buffering, audio quality issues, playback errors.",
        "examples": [
            "Songs keep pausing randomly every few seconds",
            "I'm getting an error when I try to play any song",
            "The audio quality is terrible even on high settings",
        ],
        "risk_tier": "medium",
    },
    "content_availability": {
        "label": "Content Availability",
        "definition": "Missing songs, albums, podcasts, or content not available in user's region.",
        "examples": [
            "Why was this album removed from the platform?",
            "This song isn't available in my country anymore",
            "I can't find a podcast that used to be there",
        ],
        "risk_tier": "low",
    },
    "app_technical": {
        "label": "App Crash / Technical Bug",
        "definition": "App crashes, freezes, update issues, installation problems, sync issues.",
        "examples": [
            "The app keeps crashing every time I open it",
            "After the latest update the app won't even load",
            "My playlists aren't syncing between my phone and laptop",
        ],
        "risk_tier": "medium",
    },
    "device_connectivity": {
        "label": "Device Connectivity",
        "definition": "Issues connecting to speakers, car systems, smart home devices, casting.",
        "examples": [
            "Can't connect to my Bluetooth speaker anymore",
            "The app won't connect to my car's audio system",
            "Casting to my smart TV stopped working",
        ],
        "risk_tier": "low",
    },
    "feature_feedback": {
        "label": "Feature Request / Feedback",
        "definition": "Suggestions for new features, complaints about UI changes, general feedback.",
        "examples": [
            "You should add a sleep timer feature",
            "The new UI update is terrible, bring back the old one",
            "It would be great if you could sort playlists by date added",
        ],
        "risk_tier": "low",
    },
    "other": {
        "label": "Other / General Inquiry",
        "definition": "Messages that don't fit other categories: greetings, vague complaints, off-topic.",
        "examples": [
            "Hey, quick question for you",
            "This is the worst service ever",
            "What's going on with you guys today?",
        ],
        "risk_tier": "low",
    },
}


def _build_classification_prompt(message: str) -> tuple[str, str]:
    """
    Build the system prompt and user prompt for intent classification.

    Returns (system_prompt, user_prompt).
    """
    # Build taxonomy section with examples
    taxonomy_lines = []
    for key, info in INTENT_TAXONOMY.items():
        examples_str = "\n".join(f'    - "{ex}"' for ex in info["examples"])
        taxonomy_lines.append(
            f"- **{key}**: {info['definition']}\n"
            f"  Examples:\n{examples_str}"
        )
    taxonomy_text = "\n".join(taxonomy_lines)

    system_prompt = f"""You are an intent classifier for a customer support system. 
Classify the customer message into exactly one of the following intent categories.

INTENT CATEGORIES:
{taxonomy_text}

RESPONSE FORMAT (strict — respond with ONLY this JSON, no other text):
{{"intent": "<category_key>", "confidence": <1-5>}}

Where:
- "intent" is one of: {', '.join(INTENT_TAXONOMY.keys())}
- "confidence" is 1 (very unsure) to 5 (very confident)

If the message is ambiguous between two categories, pick the more actionable one 
and lower your confidence score.
"""

    user_prompt = f'Classify this customer message:\n\n"{message}"'

    return system_prompt, user_prompt


def _parse_classification_response(response: str) -> tuple[str, float]:
    """
    Parse the LLM's JSON response into (intent, confidence).

    Handles common formatting issues: markdown code blocks, extra text.
    """
    # Strip markdown code blocks if present
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    try:
        data = json.loads(response)
        intent = data.get("intent", "other")
        confidence_raw = data.get("confidence", 3)

        # Validate intent is in taxonomy
        if intent not in INTENT_TAXONOMY:
            logger.warning("Unknown intent '%s', falling back to 'other'", intent)
            intent = "other"

        # Map 1-5 scale to 0-1
        confidence = max(0.0, min(1.0, (float(confidence_raw) - 1) / 4))
        return intent, confidence

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse classification response: %s", e)
        # Try regex fallback
        intent_match = re.search(r'"intent"\s*:\s*"(\w+)"', response)
        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', response)

        intent = intent_match.group(1) if intent_match else "other"
        confidence_raw = int(conf_match.group(1)) if conf_match else 2

        if intent not in INTENT_TAXONOMY:
            intent = "other"

        confidence = max(0.0, min(1.0, (confidence_raw - 1) / 4))
        return intent, confidence


def classify_intent(
    message: str,
    use_cache: bool = True,
) -> tuple[str, float]:
    """
    Classify a customer message into an intent category.

    Args:
        message: The customer's message text.
        use_cache: Whether to use cached LLM responses.

    Returns:
        Tuple of (intent_key, confidence_score) where confidence is 0.0–1.0.
    """
    system_prompt, user_prompt = _build_classification_prompt(message)

    # Use a hash of the message as cache key
    cache_key = f"intent_{hash(message)}"

    if use_cache:
        response = cached_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # low temperature for consistency
            max_tokens=100,
            cache_key=cache_key,
            cache_file="intent_cache.json",
        )
    else:
        response = complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=100,
        )

    return _parse_classification_response(response)


def get_intent_label(intent_key: str) -> str:
    """Get the human-readable label for an intent key."""
    return INTENT_TAXONOMY.get(intent_key, {}).get("label", "Unknown")


def get_risk_tier(intent_key: str) -> str:
    """Get the risk tier for an intent key (for escalation logic)."""
    return INTENT_TAXONOMY.get(intent_key, {}).get("risk_tier", "low")
