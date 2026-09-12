"""
LLM-as-Judge: scores drafted replies on groundedness, tone, and completeness.

Uses the rubric in eval/llm_judge_rubric.md. Ideally uses a different model
or a clearly separated persona from whatever drafted the reply, to reduce
self-preference bias.

Known limitation: even with a different model, LLM judges tend to prefer
verbose, well-structured replies. We report this and compare against
human scores (see judge_human_agreement.py).
"""

import json
import logging
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import complete, cached_complete

logger = logging.getLogger(__name__)

# Load rubric text
RUBRIC_PATH = Path(__file__).parent / "llm_judge_rubric.md"


def _load_rubric() -> str:
    """Load the rubric markdown file."""
    if RUBRIC_PATH.exists():
        return RUBRIC_PATH.read_text()
    return "(Rubric file not found — using inline scoring criteria)"


def _build_judge_prompt(
    customer_message: str,
    reply_draft: str,
    precedent_text: str,
) -> tuple[str, str]:
    """Build prompts for the LLM judge."""
    rubric = _load_rubric()

    system_prompt = f"""You are an expert quality evaluator for customer support replies.
Your task is to score a drafted reply on three dimensions using the rubric below.

RUBRIC:
{rubric}

You must be a strict, calibrated scorer. Do NOT default to high scores.
A score of 3 is "adequate" — reserve 4-5 for genuinely good replies.
Penalize generic, canned-sounding replies even if technically correct.

RESPONSE FORMAT (strict JSON only, no other text):
{{"groundedness": <1-5>, "tone": <1-5>, "completeness": <1-5>, "reasoning": "<brief explanation>"}}
"""

    user_prompt = f"""CUSTOMER MESSAGE:
"{customer_message}"

RETRIEVED PRECEDENT (used for grounding):
"{precedent_text}"

DRAFTED REPLY TO EVALUATE:
"{reply_draft}"

Score this reply on groundedness, tone/brand-voice fit, and completeness (1-5 each).
"""

    return system_prompt, user_prompt


def _parse_judge_response(response: str) -> dict:
    """Parse the judge's JSON response."""
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    try:
        data = json.loads(response)
        return {
            "groundedness": int(data.get("groundedness", 3)),
            "tone": int(data.get("tone", 3)),
            "completeness": int(data.get("completeness", 3)),
            "reasoning": data.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse judge response: %s", e)
        # Regex fallback
        scores = {}
        for dim in ["groundedness", "tone", "completeness"]:
            match = re.search(rf'"{dim}"\s*:\s*(\d)', response)
            scores[dim] = int(match.group(1)) if match else 3
        scores["reasoning"] = "Parse fallback"
        return scores


def judge_reply(
    customer_message: str,
    reply_draft: str,
    precedent_text: str,
    use_cache: bool = True,
) -> dict:
    """
    Score a drafted reply using the LLM-as-judge.

    Args:
        customer_message: The original customer message.
        reply_draft: The system's drafted reply.
        precedent_text: The top retrieved precedent's company reply.
        use_cache: Whether to use cached responses.

    Returns:
        Dict with scores: groundedness (1-5), tone (1-5),
        completeness (1-5), reasoning (str).
    """
    system_prompt, user_prompt = _build_judge_prompt(
        customer_message, reply_draft, precedent_text
    )

    cache_key = f"judge_{hash(customer_message + reply_draft)}"

    if use_cache:
        response = cached_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # very low for consistency
            max_tokens=300,
            cache_key=cache_key,
            cache_file="judge_cache.json",
        )
    else:
        response = complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=300,
        )

    return _parse_judge_response(response)


def batch_judge(
    customer_messages: list[str],
    reply_drafts: list[str],
    precedent_texts: list[str],
    use_cache: bool = True,
) -> list[dict]:
    """Score a batch of replies."""
    from tqdm import tqdm

    results = []
    for msg, reply, prec in tqdm(
        zip(customer_messages, reply_drafts, precedent_texts),
        total=len(customer_messages),
        desc="LLM Judge scoring"
    ):
        try:
            scores = judge_reply(msg, reply, prec, use_cache=use_cache)
        except Exception as e:
            logger.error("Judge failed for message: %s — %s", msg[:50], e)
            scores = {
                "groundedness": 0, "tone": 0, "completeness": 0,
                "reasoning": f"Error: {e}"
            }
        results.append(scores)
    return results
