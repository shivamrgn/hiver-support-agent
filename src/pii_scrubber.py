"""
PII scrubber for retrieved precedents.

Removes or masks personally identifiable information from historical
tweets before they are used as grounding context for reply generation.

The Kaggle dataset already masks some PII (__email__, __url__), but
retrieved precedents may still contain:
  - @mentions (real Twitter handles)
  - Phone numbers (various formats)
  - Order/case/ticket numbers
  - Names following "my name is" patterns
  - Residual emails not caught by dataset masking

Design choice: regex-based over NER (spaCy/Presidio) — simpler, faster,
and sufficient for the patterns in this dataset. A production system
would layer in NER. See DECISION_LOG.md #8.
"""

import re


# ── Pattern definitions ──────────────────────────────────────────────────

# @mentions: @username (1-15 alphanumeric + underscore chars)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]{1,15}")

# Phone numbers: various North American and international formats
_PHONE_RE = re.compile(
    r"(?<!\d)"                     # not preceded by digit
    r"(?:"
    r"\+?1?[-.\s]?"                # optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"      # area code
    r"\d{3}[-.\s]?"                # exchange
    r"\d{4}"                       # subscriber
    r")"
    r"(?!\d)"                      # not followed by digit
)

# Email addresses (catch residuals beyond __email__)
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
)

# Order / case / ticket numbers: common patterns like #1234567, case 1234567
_ORDER_RE = re.compile(
    r"(?i)(?:order|case|ticket|ref|confirmation|tracking)"
    r"[\s#:]*"
    r"[A-Z0-9-]{5,20}"
)

# Standalone long numbers that look like IDs (7+ digits)
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")

# "my name is X" pattern
_NAME_RE = re.compile(
    r"(?i)(?:my name is|i'm|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)

# URLs not already masked
_URL_RE = re.compile(
    r"https?://[^\s]+"
)


def scrub_pii(text: str) -> str:
    """
    Remove or mask PII patterns in text.

    Args:
        text: Raw text potentially containing PII.

    Returns:
        Text with PII patterns replaced by safe placeholders.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # Order matters: do specific patterns before generic ones
    result = text

    # Replace @mentions (but preserve common brand handles)
    result = _MENTION_RE.sub("@[user]", result)

    # Replace emails
    result = _EMAIL_RE.sub("[email]", result)

    # Replace URLs
    result = _URL_RE.sub("[link]", result)

    # Replace phone numbers
    result = _PHONE_RE.sub("[phone]", result)

    # Replace order/case numbers
    result = _ORDER_RE.sub("[reference_number]", result)

    # Replace "my name is X"
    result = _NAME_RE.sub(
        lambda m: m.group(0).replace(m.group(1), "[name]"), result
    )

    # Replace standalone long numbers (likely IDs)
    result = _LONG_NUMBER_RE.sub("[id_number]", result)

    # Clean up dataset-native masks for consistency
    result = result.replace("__email__", "[email]")
    result = result.replace("__url__", "[link]")

    return result


def scrub_precedent(customer_msg: str, company_reply: str) -> tuple[str, str]:
    """
    Scrub PII from a retrieved customer→company precedent pair.

    Returns:
        Tuple of (scrubbed_customer_msg, scrubbed_company_reply).
    """
    return scrub_pii(customer_msg), scrub_pii(company_reply)
