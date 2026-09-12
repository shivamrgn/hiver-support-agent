"""
Smoke tests for the end-to-end pipeline.

These tests verify that the pipeline runs without crashing and produces
well-structured output — they do NOT test correctness (that's what the
golden set evaluation is for).
"""

import sys
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pii_scrubber import scrub_pii
from src.escalation import decide_escalation
from src.intent_classifier import INTENT_TAXONOMY, _parse_classification_response


class TestPIIScrubber:
    """Tests for the PII scrubber."""

    def test_scrubs_mentions(self):
        assert "@[user]" in scrub_pii("Hey @JohnDoe can you help?")

    def test_scrubs_emails(self):
        result = scrub_pii("Email me at john@example.com please")
        assert "[email]" in result
        assert "john@example.com" not in result

    def test_scrubs_phone_numbers(self):
        result = scrub_pii("Call me at 555-123-4567")
        assert "[phone]" in result
        assert "555-123-4567" not in result

    def test_scrubs_urls(self):
        result = scrub_pii("Check https://example.com/secret for details")
        assert "[link]" in result
        assert "https://example.com" not in result

    def test_scrubs_name_pattern(self):
        result = scrub_pii("Hi, my name is John Smith and I need help")
        assert "[name]" in result

    def test_scrubs_order_numbers(self):
        result = scrub_pii("My order #12345678 hasn't arrived")
        assert "[reference_number]" in result

    def test_preserves_normal_text(self):
        text = "I can't log into my account"
        assert scrub_pii(text) == text

    def test_handles_dataset_masks(self):
        result = scrub_pii("Contact __email__ for help or visit __url__")
        assert "[email]" in result
        assert "[link]" in result
        assert "__email__" not in result

    def test_handles_empty_string(self):
        assert scrub_pii("") == ""
        assert scrub_pii(None) == ""


class TestEscalation:
    """Tests for the escalation decision logic."""

    def test_high_confidence_low_risk_auto_handles(self):
        decision, reason = decide_escalation(
            message="How do I add songs to my playlist?",
            intent_key="feature_feedback",
            confidence=0.9,
            top_similarity=0.7,
            risk_tier="low",
        )
        assert decision == "auto_handle"

    def test_low_confidence_escalates(self):
        decision, reason = decide_escalation(
            message="Something is wrong",
            intent_key="other",
            confidence=0.1,
            top_similarity=0.3,
            risk_tier="low",
        )
        assert decision == "escalate"
        assert "confidence" in reason.lower() or "similarity" in reason.lower()

    def test_legal_keyword_escalates(self):
        decision, reason = decide_escalation(
            message="I'm going to contact my lawyer about this",
            intent_key="subscription_billing",
            confidence=0.8,
            top_similarity=0.7,
            risk_tier="high",
        )
        assert decision == "escalate"
        assert "lawyer" in reason.lower()

    def test_very_short_message_escalates(self):
        decision, reason = decide_escalation(
            message="help",
            intent_key="other",
            confidence=0.5,
            top_similarity=0.5,
            risk_tier="low",
        )
        assert decision == "escalate"
        assert "short" in reason.lower()

    def test_reason_always_provided(self):
        """Every escalation decision must include a reason string."""
        for msg in ["I can't log in", "help", "cancel my subscription"]:
            _, reason = decide_escalation(
                msg, "other", 0.5, 0.5, "low"
            )
            assert isinstance(reason, str)
            assert len(reason) > 10


class TestIntentParsing:
    """Tests for intent classification response parsing."""

    def test_parses_valid_json(self):
        intent, conf = _parse_classification_response(
            '{"intent": "account_access", "confidence": 4}'
        )
        assert intent == "account_access"
        assert 0.5 < conf <= 1.0

    def test_handles_markdown_code_block(self):
        intent, conf = _parse_classification_response(
            '```json\n{"intent": "playback_streaming", "confidence": 5}\n```'
        )
        assert intent == "playback_streaming"

    def test_falls_back_on_invalid_intent(self):
        intent, _ = _parse_classification_response(
            '{"intent": "nonexistent_category", "confidence": 3}'
        )
        assert intent == "other"

    def test_falls_back_on_garbage(self):
        intent, _ = _parse_classification_response("I think this is about...")
        assert intent == "other"

    def test_taxonomy_has_required_fields(self):
        """Every intent in the taxonomy must have required fields."""
        for key, info in INTENT_TAXONOMY.items():
            assert "label" in info, f"{key} missing 'label'"
            assert "definition" in info, f"{key} missing 'definition'"
            assert "examples" in info, f"{key} missing 'examples'"
            assert "risk_tier" in info, f"{key} missing 'risk_tier'"
            assert len(info["examples"]) >= 2, f"{key} needs ≥2 examples"
