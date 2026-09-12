"""
Judge–human agreement analysis.

After the human labeler rates ~40-50 of the system's actual drafted replies,
this module computes correlation/agreement between human and LLM judge scores.

This is the required evidence that the LLM judge is a reasonable proxy for
human quality assessment — or isn't, in which case we report that honestly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_agreement(
    human_scores: list[float],
    judge_scores: list[float],
    dimension: str = "overall",
) -> dict:
    """
    Compute agreement metrics between human and judge scores.

    Args:
        human_scores: List of human-assigned scores.
        judge_scores: List of LLM-judge-assigned scores.
        dimension: Name of the dimension being compared.

    Returns:
        Dict with Pearson correlation, Spearman correlation,
        Cohen's Kappa (binned), and mean absolute error.
    """
    human = np.array(human_scores, dtype=float)
    judge = np.array(judge_scores, dtype=float)

    # Remove pairs where either is NaN
    valid = ~(np.isnan(human) | np.isnan(judge))
    human = human[valid]
    judge = judge[valid]

    if len(human) < 5:
        logger.warning(
            "Too few valid pairs (%d) for %s — skipping agreement",
            len(human), dimension
        )
        return {"error": "insufficient data", "n": len(human)}

    # Pearson correlation
    pearson_r = np.corrcoef(human, judge)[0, 1]

    # Spearman rank correlation
    from scipy import stats
    spearman_r, spearman_p = stats.spearmanr(human, judge)

    # Mean absolute error
    mae = np.mean(np.abs(human - judge))

    # Cohen's Kappa (bin 1-5 into 3 levels: low=1-2, mid=3, high=4-5)
    def _bin(scores):
        return np.where(scores <= 2, 0, np.where(scores <= 3, 1, 2))

    human_binned = _bin(human)
    judge_binned = _bin(judge)

    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(human_binned, judge_binned)

    return {
        "dimension": dimension,
        "n": len(human),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "cohens_kappa": round(float(kappa), 4),
        "mae": round(float(mae), 4),
        "human_mean": round(float(np.mean(human)), 2),
        "judge_mean": round(float(np.mean(judge)), 2),
    }


def full_agreement_report(
    human_ratings_path: str | Path,
    judge_ratings_path: str | Path,
) -> dict:
    """
    Compute agreement across all scoring dimensions.

    Expects CSVs with columns: example_id, groundedness, tone, completeness.

    Returns dict with per-dimension agreement + overall agreement.
    """
    human_df = pd.read_csv(human_ratings_path)
    judge_df = pd.read_csv(judge_ratings_path)

    # Merge on example_id
    merged = human_df.merge(
        judge_df, on="example_id", suffixes=("_human", "_judge")
    )

    results = {}
    for dim in ["groundedness", "tone", "completeness"]:
        h_col = f"{dim}_human"
        j_col = f"{dim}_judge"
        if h_col in merged.columns and j_col in merged.columns:
            results[dim] = compute_agreement(
                merged[h_col].tolist(),
                merged[j_col].tolist(),
                dimension=dim,
            )

    # Overall (average across dimensions)
    h_overall = merged[
        [c for c in merged.columns if c.endswith("_human") and c != "example_id"]
    ].mean(axis=1).tolist()
    j_overall = merged[
        [c for c in merged.columns if c.endswith("_judge") and c != "example_id"]
    ].mean(axis=1).tolist()
    results["overall"] = compute_agreement(h_overall, j_overall, "overall")

    return results
