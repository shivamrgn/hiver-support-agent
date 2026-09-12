"""
Evaluation runner: single entrypoint for all metrics.

Usage:
    python eval/run_eval.py           # use cached LLM outputs (fast, <5 min)
    python eval/run_eval.py --fresh   # call LLM APIs (slower, needs API key)

Outputs:
    results/metrics.json          — all computed metrics
    results/eval_examples.csv     — per-example predictions + judge scores
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.pipeline import end_to_end
from src.baselines import (
    MajorityClassBaseline, TfidfLogregBaseline,
    canned_reply_baseline, zero_shot_llm_baseline,
    always_escalate, always_auto_handle, keyword_only_escalation,
)
from eval.metrics import intent_metrics, escalation_metrics, batch_groundedness
from eval.llm_judge import batch_judge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_golden_set() -> pd.DataFrame:
    """Load the hand-labeled golden set."""
    labeled_path = DATA_DIR / "golden_set_labeled.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {labeled_path}. "
            "Run the labeling tool first: make label"
        )

    df = pd.read_csv(labeled_path)

    # Filter to fully labeled examples
    df = df[
        (df["intent_label"].astype(str).str.strip() != "") &
        (df["escalation_label"].astype(str).str.strip() != "")
    ].copy()

    logger.info("Loaded %d labeled examples from golden set", len(df))
    return df


def run_pipeline_on_golden(
    golden: pd.DataFrame,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run the full pipeline on all golden set examples."""
    results = []

    for _, row in tqdm(golden.iterrows(), total=len(golden),
                       desc="Running pipeline"):
        message = str(row["text"])
        try:
            output = end_to_end(message, use_cache=use_cache)
        except Exception as e:
            logger.error("Pipeline failed for: %s — %s", message[:50], e)
            output = {
                "intent": "other", "confidence": 0.0,
                "reply_draft": "", "escalation_decision": "escalate",
                "escalation_reason": f"Error: {e}",
                "retrieved_precedents": [], "retrieval_scores": [],
            }

        results.append({
            "tweet_id": row.get("tweet_id", ""),
            "text": message,
            "true_intent": row["intent_label"],
            "pred_intent": output["intent"],
            "confidence": output["confidence"],
            "true_escalation": row["escalation_label"],
            "pred_escalation": output["escalation_decision"],
            "escalation_reason": output["escalation_reason"],
            "reply_draft": output["reply_draft"],
            "top_precedent": (
                output["retrieved_precedents"][0]["company_text"]
                if output.get("retrieved_precedents") else ""
            ),
            "top_similarity": (
                output["retrieval_scores"][0]
                if output.get("retrieval_scores") else 0.0
            ),
            "sample_bucket": row.get("sample_bucket", ""),
        })

    return pd.DataFrame(results)


def run_baselines(golden: pd.DataFrame, use_cache: bool = True) -> dict:
    """Run all baselines on the golden set and compute metrics."""
    messages = golden["text"].astype(str).tolist()
    true_intents = golden["intent_label"].astype(str).tolist()
    true_escalations = golden["escalation_label"].astype(str).tolist()

    baseline_results = {}

    # ── Intent baselines ─────────────────────────────────────────────
    # Majority class
    majority = MajorityClassBaseline()
    majority.fit(true_intents)
    majority_preds = majority.predict(messages)
    baseline_results["intent_majority"] = intent_metrics(true_intents, majority_preds)

    # TF-IDF + LogReg (trained on taxonomy examples)
    tfidf = TfidfLogregBaseline()
    tfidf.fit_from_taxonomy()
    tfidf_preds = tfidf.predict(messages)
    baseline_results["intent_tfidf_logreg"] = intent_metrics(true_intents, tfidf_preds)

    # ── Escalation baselines ─────────────────────────────────────────
    always_esc_preds = [always_escalate(m)[0] for m in messages]
    baseline_results["escalation_always_escalate"] = escalation_metrics(
        true_escalations, always_esc_preds
    )

    always_auto_preds = [always_auto_handle(m)[0] for m in messages]
    baseline_results["escalation_always_auto"] = escalation_metrics(
        true_escalations, always_auto_preds
    )

    keyword_preds = [keyword_only_escalation(m)[0] for m in messages]
    baseline_results["escalation_keyword_only"] = escalation_metrics(
        true_escalations, keyword_preds
    )

    return baseline_results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--fresh", action="store_true",
                        help="Call LLM APIs fresh (don't use cache)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge scoring (faster)")
    args = parser.parse_args()

    use_cache = not args.fresh
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load golden set
    golden = load_golden_set()

    # 2. Run pipeline on golden set
    logger.info("=== Running Pipeline ===")
    eval_df = run_pipeline_on_golden(golden, use_cache=use_cache)

    # 3. Compute main system metrics
    logger.info("=== Computing Metrics ===")
    metrics = {}

    # Intent classification
    metrics["intent"] = intent_metrics(
        eval_df["true_intent"].tolist(),
        eval_df["pred_intent"].tolist(),
    )
    logger.info("Intent accuracy: %.1f%%, macro-F1: %.3f",
                metrics["intent"]["accuracy"] * 100,
                metrics["intent"]["macro_f1"])

    # Escalation
    metrics["escalation"] = escalation_metrics(
        eval_df["true_escalation"].tolist(),
        eval_df["pred_escalation"].tolist(),
    )
    logger.info("Escalation accuracy: %.1f%%, escalate-F1: %.3f",
                metrics["escalation"]["accuracy"] * 100,
                metrics["escalation"]["escalate_f1"])

    # Groundedness
    valid_replies = eval_df[
        (eval_df["reply_draft"].str.strip() != "") &
        (eval_df["top_precedent"].str.strip() != "")
    ]
    if len(valid_replies) > 0:
        metrics["groundedness"] = batch_groundedness(
            valid_replies["reply_draft"].tolist(),
            valid_replies["top_precedent"].tolist(),
        )
        # Remove per-example scores from summary (too verbose)
        del metrics["groundedness"]["scores"]
        logger.info("Groundedness (ROUGE-L): mean=%.3f",
                     metrics["groundedness"]["mean"])

    # 4. LLM Judge scoring
    if not args.skip_judge:
        logger.info("=== Running LLM Judge ===")
        judge_scores = batch_judge(
            eval_df["text"].tolist(),
            eval_df["reply_draft"].tolist(),
            eval_df["top_precedent"].tolist(),
            use_cache=use_cache,
        )

        # Add judge scores to eval_df
        for i, scores in enumerate(judge_scores):
            eval_df.at[i, "judge_groundedness"] = scores["groundedness"]
            eval_df.at[i, "judge_tone"] = scores["tone"]
            eval_df.at[i, "judge_completeness"] = scores["completeness"]
            eval_df.at[i, "judge_reasoning"] = scores["reasoning"]

        metrics["llm_judge"] = {
            "mean_groundedness": round(
                eval_df["judge_groundedness"].mean(), 2),
            "mean_tone": round(eval_df["judge_tone"].mean(), 2),
            "mean_completeness": round(
                eval_df["judge_completeness"].mean(), 2),
            "mean_overall": round(
                eval_df[["judge_groundedness", "judge_tone",
                         "judge_completeness"]].mean(axis=1).mean(), 2),
        }
        logger.info("LLM Judge mean scores — G:%.1f T:%.1f C:%.1f",
                     metrics["llm_judge"]["mean_groundedness"],
                     metrics["llm_judge"]["mean_tone"],
                     metrics["llm_judge"]["mean_completeness"])

    # 5. Run baselines
    logger.info("=== Running Baselines ===")
    metrics["baselines"] = run_baselines(golden, use_cache=use_cache)

    # 6. Save results
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    eval_path = RESULTS_DIR / "eval_examples.csv"
    eval_df.to_csv(eval_path, index=False)
    logger.info("Per-example results saved to %s", eval_path)

    # 7. Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Golden set size: {len(golden)}")
    print(f"\nIntent Classification:")
    print(f"  System:         accuracy={metrics['intent']['accuracy']:.1%}  "
          f"macro-F1={metrics['intent']['macro_f1']:.3f}")
    if "intent_majority" in metrics.get("baselines", {}):
        bm = metrics["baselines"]["intent_majority"]
        print(f"  Majority class: accuracy={bm['accuracy']:.1%}  "
              f"macro-F1={bm['macro_f1']:.3f}")
    if "intent_tfidf_logreg" in metrics.get("baselines", {}):
        bt = metrics["baselines"]["intent_tfidf_logreg"]
        print(f"  TF-IDF+LogReg:  accuracy={bt['accuracy']:.1%}  "
              f"macro-F1={bt['macro_f1']:.3f}")
    print(f"\nEscalation:")
    print(f"  System:           accuracy={metrics['escalation']['accuracy']:.1%}  "
          f"escalate-F1={metrics['escalation']['escalate_f1']:.3f}  "
          f"false-auto={metrics['escalation']['false_auto_handle']}")
    if "groundedness" in metrics:
        print(f"\nGroundedness (ROUGE-L): mean={metrics['groundedness']['mean']:.3f}")
    if "llm_judge" in metrics:
        j = metrics["llm_judge"]
        print(f"\nLLM Judge (1-5 scale):")
        print(f"  Groundedness: {j['mean_groundedness']:.1f}  "
              f"Tone: {j['mean_tone']:.1f}  "
              f"Completeness: {j['mean_completeness']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
