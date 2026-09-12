"""
Reply generator: drafts a response grounded in retrieved precedents.

The LLM is prompted to paraphrase the resolution pattern from precedents,
not copy them verbatim. The draft should match the brand's tone and voice.

All precedents passed to this module must already be PII-scrubbed
(the retrieval module handles this).
"""

import logging

from src.llm_client import complete, cached_complete

logger = logging.getLogger(__name__)


def _build_reply_prompt(
    customer_message: str,
    intent_label: str,
    precedents: list[dict],
) -> tuple[str, str]:
    """Build the system and user prompts for reply generation."""

    # Format precedents
    precedent_text = ""
    for i, p in enumerate(precedents, 1):
        precedent_text += (
            f"\n--- Precedent {i} (similarity: {p['similarity']:.2f}) ---\n"
            f"Customer said: {p['customer_text']}\n"
            f"Brand replied: {p['company_text']}\n"
        )

    system_prompt = """You are a helpful customer support agent. Your task is to draft a reply 
to a customer message, grounded in how similar issues have been resolved before.

CRITICAL RULES:
1. PARAPHRASE the resolution pattern from the precedents — NEVER copy them word-for-word.
2. Address the SPECIFIC issue the customer describes, not a generic version of it.
3. Match the brand's tone: professional, empathetic, and concise (Twitter-style, under 280 chars if possible).
4. If the precedents suggest sending the customer to DM for account-specific help, do the same.
5. Include specific next steps when possible.
6. Do NOT reference other customers or say "we've seen this before."
7. Do NOT include any personally identifiable information.
8. Keep the reply focused and actionable — avoid filler phrases.

You are NOT actually resolving the issue — you are drafting a response for a human agent to review."""

    user_prompt = f"""CUSTOMER MESSAGE:
"{customer_message}"

CLASSIFIED INTENT: {intent_label}

SIMILAR RESOLVED CASES (for grounding — paraphrase, don't copy):
{precedent_text}

Draft a reply to this customer. Be empathetic, specific, and action-oriented."""

    return system_prompt, user_prompt


def generate_reply(
    customer_message: str,
    intent_label: str,
    precedents: list[dict],
    use_cache: bool = True,
) -> str:
    """
    Generate a reply draft grounded in retrieved precedents.

    Args:
        customer_message: The customer's incoming message.
        intent_label: The classified intent (human-readable label).
        precedents: List of PII-scrubbed precedent dicts from retrieval.
        use_cache: Whether to use cached LLM responses.

    Returns:
        The drafted reply text.
    """
    system_prompt, user_prompt = _build_reply_prompt(
        customer_message, intent_label, precedents
    )

    cache_key = f"reply_{hash(customer_message)}"

    if use_cache:
        response = cached_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,  # slightly creative but grounded
            max_tokens=300,
            cache_key=cache_key,
            cache_file="reply_cache.json",
        )
    else:
        response = complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=300,
        )

    # Clean up: remove any quotes wrapping the response
    reply = response.strip().strip('"').strip("'")
    return reply
