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
    "compute_containers": {
        "label": "Compute & Containers",
        "definition": "Issues with Virtual Machines, App Services, AKS (Kubernetes), Cloud Services, or scaling.",
        "examples": [
            "Getting the following error when trying to create a cloud service",
            "What is the min VM size required to apply PowerShell DSC",
            "az aks create fails",
        ],
        "risk_tier": "medium",
    },
    "database_storage": {
        "label": "Database & Storage",
        "definition": "Issues with Azure SQL, CosmosDB, Storage Accounts, Redis, or data loss.",
        "examples": [
            "azure db having problems in west EU again?",
            "I have 3 instance on Azure Sqlsever... one of them can not connect",
            "My blob storage is returning 403 forbidden",
        ],
        "risk_tier": "medium",
    },
    "identity_security_network": {
        "label": "Identity, Security & Network",
        "definition": "Azure AD, MFA, Intune, VPN, Certificates, VNet, DNS resolution, and RBAC issues.",
        "examples": [
            "error al conectar la vpn en azure, No se encuentra un certificado",
            "Looking at Azure AD Premium (to use conditional rules for MFA.)",
            "Title is 401 Error From Corporate Network",
        ],
        "risk_tier": "high",
    },
    "billing_subscription": {
        "label": "Billing & Subscription",
        "definition": "Payment failures, credit limits, subscriptions disabled, pricing questions, trial issues.",
        "examples": [
            "Am getting a warning 'That my service will be disabled...Payment has not been received'",
            "Other than the $150 monthly credit limit, I didn't think there were any subscriptions",
            "Why was I charged for a free trial?",
        ],
        "risk_tier": "high",
    },
    "portal_tools": {
        "label": "Portal & Tools",
        "definition": "Azure Portal UI bugs, Cloud Shell, Azure CLI, deployment errors, Azure PowerShell.",
        "examples": [
            "buttons of cloud shell (change powershell/cli, help, settings) don't work in FF",
            "Who is in charge of the new Azure portal for Intune? I want to yell at you...",
            "But I couldn't check App logs in 'Continuous Export'.",
        ],
        "risk_tier": "low",
    },
    "other": {
        "label": "Other / General Inquiry",
        "definition": "Vague complaints, greetings, non-English messages, feedback without specific service mention.",
        "examples": [
            "And trying to get support is way more difficult then it should be",
            "Has this been resolved?",
            "OK",
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
