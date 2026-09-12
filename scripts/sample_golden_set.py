"""
Golden set sampler: create a stratified sample of customer messages
for human labeling.

Sampling strategy:
  - ~60% random sample across all messages
  - ~20% deliberately short messages (<30 chars) — these are often ambiguous
  - ~10% messages with strong negative sentiment (anger/sarcasm)
  - ~10% messages with keywords suggesting complex/multi-issue cases

Target: 200 examples (allows some discard during labeling, keeping 150–200).

Usage:
    python scripts/sample_golden_set.py
    python scripts/sample_golden_set.py --n 250
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Patterns for stratification
_NEGATIVE_RE = re.compile(
    r"(?i)\b(worst|terrible|horrible|disgusting|scam|fraud|useless|"
    r"unacceptable|pathetic|incompetent|never\s+again|switching|hate)\b"
)

_COMPLEX_RE = re.compile(
    r"(?i)\b(also|another\s+issue|on\s+top\s+of|additionally|plus|"
    r"and\s+also|multiple|several\s+problems|cancel.*refund|"
    r"lawsuit|lawyer|attorney)\b"
)


def sample_golden_set(
    subsample_path: Path = DATA_DIR / "brand_subsample.csv",
    output_path: Path = DATA_DIR / "golden_set_unlabeled.csv",
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a stratified sample of customer messages for human labeling.

    Args:
        subsample_path: Path to the brand subsample CSV.
        output_path: Where to write the unlabeled golden set.
        n: Target number of examples.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the sampled messages.
    """
    rng = np.random.RandomState(seed)

    df = pd.read_csv(subsample_path)
    logger.info("Loaded %d rows from subsample", len(df))

    # Filter to inbound (customer) messages only
    if "inbound" in df.columns:
        # Handle both string and bool types
        if df["inbound"].dtype == object:
            customers = df[df["inbound"].astype(str).str.lower() == "true"].copy()
        else:
            customers = df[df["inbound"] == True].copy()
    else:
        customers = df.copy()

    customers = customers[customers["text"].notna()].copy()
    customers["text"] = customers["text"].astype(str)
    logger.info("Customer messages available for sampling: %d", len(customers))

    # ── Stratification buckets ───────────────────────────────────────
    n_random = int(n * 0.60)
    n_short = int(n * 0.20)
    n_negative = int(n * 0.10)
    n_complex = n - n_random - n_short - n_negative  # remainder

    sampled_ids = set()
    samples = []

    # Bucket 1: Short messages (<30 chars)
    short = customers[customers["text"].str.len() < 30]
    short_sample = short.sample(min(n_short, len(short)), random_state=rng)
    samples.append(short_sample)
    sampled_ids.update(short_sample["tweet_id"])
    logger.info("Short messages sampled: %d/%d target", len(short_sample), n_short)

    # Bucket 2: Negative sentiment
    remaining = customers[~customers["tweet_id"].isin(sampled_ids)]
    negative = remaining[remaining["text"].apply(lambda x: bool(_NEGATIVE_RE.search(x)))]
    neg_sample = negative.sample(min(n_negative, len(negative)), random_state=rng)
    samples.append(neg_sample)
    sampled_ids.update(neg_sample["tweet_id"])
    logger.info("Negative messages sampled: %d/%d target", len(neg_sample), n_negative)

    # Bucket 3: Complex / multi-issue
    remaining = customers[~customers["tweet_id"].isin(sampled_ids)]
    complex_msgs = remaining[remaining["text"].apply(lambda x: bool(_COMPLEX_RE.search(x)))]
    complex_sample = complex_msgs.sample(min(n_complex, len(complex_msgs)), random_state=rng)
    samples.append(complex_sample)
    sampled_ids.update(complex_sample["tweet_id"])
    logger.info("Complex messages sampled: %d/%d target", len(complex_sample), n_complex)

    # Bucket 4: Random from remainder
    remaining = customers[~customers["tweet_id"].isin(sampled_ids)]
    random_sample = remaining.sample(min(n_random, len(remaining)), random_state=rng)
    samples.append(random_sample)
    logger.info("Random messages sampled: %d/%d target", len(random_sample), n_random)

    # Combine and shuffle
    golden = pd.concat(samples, ignore_index=True)
    golden = golden.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Add labeling columns (to be filled by human)
    golden["intent_label"] = ""
    golden["escalation_label"] = ""  # "auto_handle" or "escalate"
    golden["notes"] = ""
    golden["sample_bucket"] = ""  # will be filled below

    # Tag sample buckets for transparency
    short_ids = set(short_sample["tweet_id"])
    neg_ids = set(neg_sample["tweet_id"])
    complex_ids = set(complex_sample["tweet_id"])
    golden.loc[golden["tweet_id"].isin(short_ids), "sample_bucket"] = "short"
    golden.loc[golden["tweet_id"].isin(neg_ids), "sample_bucket"] = "negative"
    golden.loc[golden["tweet_id"].isin(complex_ids), "sample_bucket"] = "complex"
    golden.loc[golden["sample_bucket"] == "", "sample_bucket"] = "random"

    # Save
    golden.to_csv(output_path, index=False)
    logger.info(
        "\n=== Golden Set Sampling Complete ===\n"
        "Total sampled: %d\n"
        "  Random: %d\n"
        "  Short (<30 chars): %d\n"
        "  Negative sentiment: %d\n"
        "  Complex/multi-issue: %d\n"
        "Output: %s",
        len(golden),
        len(random_sample), len(short_sample),
        len(neg_sample), len(complex_sample),
        output_path,
    )

    return golden


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample golden set for labeling")
    parser.add_argument("--n", type=int, default=200, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sample_golden_set(n=args.n, seed=args.seed)
