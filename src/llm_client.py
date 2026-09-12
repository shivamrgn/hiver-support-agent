"""
Thin multi-provider LLM client wrapper.

Auto-detects the first available provider from environment variables:
  OPENAI_API_KEY → OpenAI
  ANTHROPIC_API_KEY → Anthropic
  GOOGLE_API_KEY → Google Gemini

Override with LLM_PROVIDER=openai|anthropic|google in .env.
"""

import os
import time
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Default models per provider (overridable via env)
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "google": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


def _detect_provider() -> str:
    """Return the first available provider based on env vars."""
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced in ("openai", "anthropic", "google"):
        return forced

    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"

    raise RuntimeError(
        "No LLM API key found. Set at least one of: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY in .env"
    )


def _call_openai(prompt: str, system_prompt: str, temperature: float,
                 max_tokens: int, model: str) -> str:
    """Call OpenAI's chat completions API."""
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, system_prompt: str, temperature: float,
                    max_tokens: int, model: str) -> str:
    """Call Anthropic's messages API."""
    import anthropic
    client = anthropic.Anthropic()

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_google(prompt: str, system_prompt: str, temperature: float,
                 max_tokens: int, model: str) -> str:
    """Call Google Gemini API via the google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


_CALLERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
}


def complete(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    Args:
        prompt: The user message / main prompt.
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in the response.
        provider: Force a specific provider; auto-detected if None.
        model: Force a specific model; uses provider default if None.

    Returns:
        The LLM's response as a plain string.

    Raises:
        RuntimeError: If no API key is configured or all retries fail.
    """
    provider = provider or _detect_provider()
    model = model or DEFAULT_MODELS[provider]
    caller = _CALLERS[provider]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = caller(prompt, system_prompt, temperature, max_tokens, model)
            return result
        except Exception as e:
            last_error = e
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d, provider=%s): %s. "
                "Retrying in %ds...",
                attempt + 1, MAX_RETRIES, provider, e, wait
            )
            time.sleep(wait)

    raise RuntimeError(
        f"All {MAX_RETRIES} LLM call attempts failed. Last error: {last_error}"
    )


# ── Caching helpers ──────────────────────────────────────────────────────

def cached_complete(
    prompt: str,
    cache_key: str,
    cache_dir: str | Path = "results/cache",
    cache_file: str = "llm_cache.json",
    **kwargs,
) -> str:
    """
    Like complete(), but checks a JSON cache file first.

    If cache_key exists in the cache, return the cached result without
    calling the LLM. Otherwise, call the LLM, cache the result, and return.

    This is the mechanism behind the fast (<15 min) reproduction path:
    headline numbers are backed by cached outputs, and --fresh regenerates
    them from scratch.
    """
    cache_path = Path(cache_dir) / cache_file
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}

    if cache_key in cache:
        return cache[cache_key]

    # Cache miss — call LLM
    result = complete(prompt, **kwargs)

    # Save to cache
    cache[cache_key] = result
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    return result
