"""
Data preparation: load, reconstruct threads, analyze brands, subsample.

Usage:
    python -m src.data_prep                   # full pipeline
    python -m src.data_prep --brand SpotifyCares  # force a specific brand
    python -m src.data_prep --skip-analysis   # skip brand comparison (reuse existing subsample)

Expects data/twcs.csv (the full Kaggle dataset). Outputs:
    data/brand_subsample.csv   — all tweets for the chosen brand
    data/resolved_pairs.csv    — customer→company resolved pairs for retrieval
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data"
TWCS_PATH = DATA_DIR / "twcs.csv"

# Expected columns per the Kaggle dataset description
EXPECTED_COLS = {
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
}

# Positive sentiment patterns suggesting a resolved interaction
_POSITIVE_RE = re.compile(
    r"(?i)\b(thanks?|thank\s*you|thx|awesome|great|perfect|solved|fixed|works?\s*now|appreciate)\b"
)

# Negative sentiment patterns suggesting an unresolved interaction
_NEGATIVE_RE = re.compile(
    r"(?i)\b(still\s*(not|broken|down|can'?t)|worst|terrible|horrible|"
    r"unacceptable|useless|disgusting|scam|lawsuit|attorney|lawyer|"
    r"not\s*help|didn'?t\s*(help|work|fix)|same\s*(issue|problem))\b"
)


def load_dataset(path: Path = TWCS_PATH) -> pd.DataFrame:
    """Load the TWCS dataset and verify column schema."""
    logger.info("Loading dataset from %s ...", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from:\n"
            "  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
            "and place twcs.csv in the data/ directory."
        )

    df = pd.read_csv(path)
    actual_cols = set(df.columns)
    missing = EXPECTED_COLS - actual_cols
    if missing:
        logger.warning(
            "Missing expected columns: %s. Actual columns: %s",
            missing, list(df.columns)
        )
    else:
        logger.info(
            "Schema verified. %d rows, columns: %s",
            len(df), list(df.columns)
        )

    # Normalize inbound column (might be True/False strings)
    if df["inbound"].dtype == object:
        df["inbound"] = df["inbound"].map({"True": True, "False": False})

    # Convert tweet IDs to string for consistent joining
    for col in ["tweet_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].apply(
            lambda x: "" if pd.isna(x) or str(x) == "nan" else (str(x)[:-2] if str(x).endswith(".0") else str(x))
        )

    return df


def identify_brands(df: pd.DataFrame) -> list[str]:
    """Identify brand/company author_ids (those that post non-inbound tweets)."""
    company_tweets = df[df["inbound"] == False]
    brands = company_tweets["author_id"].unique().tolist()
    logger.info("Found %d unique brand accounts", len(brands))
    return brands


def _is_english_heuristic(text: str) -> bool:
    """Simple heuristic: >80% ASCII characters suggests English."""
    if not isinstance(text, str) or len(text) == 0:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) > 0.8


def analyze_brands(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Compute key metrics for each brand to inform brand selection.

    Returns a DataFrame with columns:
        brand, inbound_count, company_reply_count, resolved_pct,
        english_pct, avg_thread_length
    """
    brands = identify_brands(df)
    inbound = df[df["inbound"] == True]
    outbound = df[df["inbound"] == False]

    # Map: which inbound tweets got a company reply?
    # in_response_to_tweet_id on outbound tweets points to the inbound tweet
    replied_to = set(outbound["in_response_to_tweet_id"].dropna().unique())

    results = []
    for brand in tqdm(brands, desc="Analyzing brands"):
        # Company tweets from this brand
        brand_out = outbound[outbound["author_id"] == brand]

        # Inbound tweets that this brand replied to
        brand_replied_ids = set(brand_out["in_response_to_tweet_id"].dropna().unique())

        # All inbound tweets directed at this brand (they mention the brand
        # or are in threads where the brand participates)
        # Simpler: inbound tweets that have a response from this brand
        brand_inbound = inbound[inbound["tweet_id"].isin(brand_replied_ids)]

        inbound_count = len(brand_inbound)
        if inbound_count < 100:
            continue  # skip brands with very few interactions

        # English percentage
        sample_texts = brand_inbound["text"].dropna().sample(
            min(200, inbound_count), random_state=42
        )
        eng_pct = sample_texts.apply(_is_english_heuristic).mean()

        # Resolved percentage: brand replied AND no angry follow-up
        # (simplified: brand replied at all)
        resolved_pct = len(brand_replied_ids) / max(inbound_count, 1)

        # Average thread depth (how many turns in conversations)
        # Approximate: count response chains
        avg_thread_len = 1.0  # default
        if len(brand_out) > 0:
            has_followup = brand_out["response_tweet_id"].apply(
                lambda x: len(str(x).split(",")) if pd.notna(x) and str(x) else 0
            )
            avg_thread_len = 1 + has_followup.mean()

        results.append({
            "brand": brand,
            "inbound_count": inbound_count,
            "company_reply_count": len(brand_out),
            "resolved_pct": round(resolved_pct * 100, 1),
            "english_pct": round(eng_pct * 100, 1),
            "avg_thread_length": round(avg_thread_len, 2),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("inbound_count", ascending=False)

    logger.info("\n=== Brand Analysis (Top %d) ===", top_n)
    logger.info("\n%s", results_df.head(top_n).to_string(index=False))

    return results_df


def select_brand(analysis: pd.DataFrame) -> str:
    """
    Select the best brand for this project based on the analysis.

    Criteria:
        1. English share > 85%
        2. Inbound volume between 2,000 and 50,000 (manageable subsample)
        3. Highest resolved percentage among qualifying brands
    """
    candidates = analysis[
        (analysis["english_pct"] > 85) &
        (analysis["inbound_count"] >= 2000) &
        (analysis["inbound_count"] <= 50000)
    ].copy()

    if len(candidates) == 0:
        # Relax constraints
        candidates = analysis[analysis["english_pct"] > 70].copy()
        logger.warning("Relaxed selection criteria — fewer qualifying brands")

    # Sort by resolved_pct descending, then inbound_count descending
    candidates = candidates.sort_values(
        ["resolved_pct", "inbound_count"], ascending=[False, False]
    )

    selected = candidates.iloc[0]["brand"]
    logger.info("Selected brand: %s", selected)
    return selected


def extract_brand_subsample(
    df: pd.DataFrame,
    brand: str,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """
    Extract all tweets for the chosen brand: both inbound customer tweets
    directed at this brand and the brand's own outbound replies.

    Caps at max_rows to keep the subsample manageable.
    """
    outbound = df[(df["inbound"] == False) & (df["author_id"] == brand)]

    # Inbound tweets that this brand replied to
    replied_to_ids = set(outbound["in_response_to_tweet_id"].dropna().unique())
    inbound = df[(df["inbound"] == True) & (df["tweet_id"].isin(replied_to_ids))]

    # Also include inbound tweets that are in response to the brand's tweets
    # (follow-ups from customers)
    brand_tweet_ids = set(outbound["tweet_id"].unique())
    followups = df[
        (df["inbound"] == True) &
        (df["in_response_to_tweet_id"].isin(brand_tweet_ids))
    ]

    # Combine all relevant tweets
    all_ids = (
        set(outbound["tweet_id"]) |
        set(inbound["tweet_id"]) |
        set(followups["tweet_id"])
    )
    subsample = df[df["tweet_id"].isin(all_ids)].copy()

    # Cap size if needed (keep balanced inbound/outbound)
    if len(subsample) > max_rows:
        # Sample proportionally
        subsample = subsample.sample(max_rows, random_state=42)

    logger.info(
        "Brand subsample for '%s': %d tweets (%d inbound, %d outbound)",
        brand, len(subsample),
        (subsample["inbound"] == True).sum(),
        (subsample["inbound"] == False).sum(),
    )
    return subsample


def extract_resolved_pairs(
    df: pd.DataFrame,
    brand: str,
) -> pd.DataFrame:
    """
    Extract customer→company resolved pairs for the retrieval corpus.

    A pair is considered "resolved" if:
        1. A customer (inbound) tweet exists
        2. The brand replied to it (outbound)
        3. Either the thread ends after the brand's reply, OR the
           customer's follow-up has positive sentiment (thanks, etc.)
           — NOT a repeated complaint.

    Returns a DataFrame with columns:
        customer_tweet_id, customer_text, company_tweet_id, company_text,
        is_resolved (bool)
    """
    outbound = df[(df["inbound"] == False) & (df["author_id"] == brand)]
    inbound = df[df["inbound"] == True]

    # Build lookup: tweet_id → row
    tweet_lookup = df.set_index("tweet_id")

    pairs = []
    for _, reply in tqdm(outbound.iterrows(), total=len(outbound),
                         desc="Extracting resolved pairs"):
        parent_id = reply["in_response_to_tweet_id"]
        if not parent_id or parent_id == "":
            continue

        # Find the customer message this is replying to
        if parent_id not in tweet_lookup.index:
            continue

        parent = tweet_lookup.loc[parent_id]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]

        # Check if the parent was inbound (customer message)
        if parent.get("inbound") != True:
            continue

        customer_text = str(parent.get("text", ""))
        company_text = str(reply.get("text", ""))

        if not customer_text.strip() or not company_text.strip():
            continue

        # Check for follow-ups to the brand's reply
        reply_id = str(reply["tweet_id"])
        followups = df[df["in_response_to_tweet_id"] == reply_id]

        is_resolved = True
        if len(followups) > 0:
            # Check if follow-ups are negative (unresolved)
            for _, fu in followups.iterrows():
                fu_text = str(fu.get("text", ""))
                if _NEGATIVE_RE.search(fu_text):
                    is_resolved = False
                    break

        pairs.append({
            "customer_tweet_id": parent_id,
            "customer_text": customer_text,
            "company_tweet_id": reply["tweet_id"],
            "company_text": company_text,
            "is_resolved": is_resolved,
        })

    pairs_df = pd.DataFrame(pairs)

    # Keep only resolved pairs
    resolved = pairs_df[pairs_df["is_resolved"] == True].copy()
    logger.info(
        "Extracted %d total pairs, %d resolved (%.1f%%)",
        len(pairs_df), len(resolved),
        len(resolved) / max(len(pairs_df), 1) * 100,
    )
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Prepare brand data subsample")
    parser.add_argument("--brand", type=str, default=None,
                        help="Force a specific brand (skip analysis)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip brand comparison analysis")
    parser.add_argument("--max-rows", type=int, default=5000,
                        help="Max rows in the brand subsample")
    args = parser.parse_args()

    # Load full dataset
    df = load_dataset()

    # Brand analysis and selection
    if args.brand:
        brand = args.brand
        logger.info("Using forced brand: %s", brand)
    elif args.skip_analysis:
        brand = "SpotifyCares"  # default fallback
        logger.info("Skipping analysis, using default brand: %s", brand)
    else:
        analysis = analyze_brands(df)
        analysis.to_csv(DATA_DIR / "brand_analysis.csv", index=False)
        brand = select_brand(analysis)

    # Extract subsample
    subsample = extract_brand_subsample(df, brand, max_rows=args.max_rows)
    subsample_path = DATA_DIR / "brand_subsample.csv"
    subsample.to_csv(subsample_path, index=False)
    logger.info("Saved brand subsample to %s", subsample_path)

    # Extract resolved pairs
    resolved = extract_resolved_pairs(df, brand)
    resolved_path = DATA_DIR / "resolved_pairs.csv"
    resolved.to_csv(resolved_path, index=False)
    logger.info("Saved %d resolved pairs to %s", len(resolved), resolved_path)

    # Summary
    logger.info("\n=== Phase 1 Complete ===")
    logger.info("Brand: %s", brand)
    logger.info("Subsample: %d tweets", len(subsample))
    logger.info("Resolved pairs: %d (retrieval corpus)", len(resolved))
    logger.info("Files: %s, %s", subsample_path, resolved_path)


if __name__ == "__main__":
    main()
