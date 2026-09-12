# Hiver SDE Intern — AI Support Agent: Evaluation Report

## 1. Problem Framing

### What "good" means for this brand

*[To be filled after Phase 1 brand selection and Phase 6 evaluation]*

A good customer support agent for [brand] must:
- Correctly understand what the customer is asking about (intent classification)
- Propose a response that's grounded in how the brand actually handles these issues (not hallucinated advice)
- Know when to auto-handle (low-risk, clear issue, strong precedent match) vs. escalate to a human (high-risk, ambiguous, novel issue)

The asymmetric cost structure is critical: a false auto-handle (sending a robotic reply when a human should intervene) is much worse than a false escalate (routing to a human unnecessarily). We optimize for **escalation recall** over escalation precision.

### What was deliberately not built

- **Multi-turn conversation management**: The agent handles single messages, not full conversation threads. A production system would track conversation state.
- **Actual issue resolution**: The agent drafts a reply; it doesn't execute actions (refunds, password resets, account changes).
- **Real-time learning**: The retrieval corpus is static. A production system would continuously update with new resolved cases.
- **Multilingual support**: English only (this brand's customer base is predominantly English-speaking).
- **Full PII protection**: Regex-based PII scrubbing is a best-effort heuristic, not a production-grade solution. See failure modes.
- **Fine-tuned models**: We use LLM few-shot classification and generation, not fine-tuned models. With a labeled training set (like the golden set), fine-tuning would likely improve classification.

---

## 2. Results vs. Baselines

*[To be filled after Phase 6 evaluation]*

### Intent Classification

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| **Our system** (LLM few-shot) | TBD | TBD |
| TF-IDF + Logistic Regression | TBD | TBD |
| Majority class (always predict most common) | TBD | TBD |

### Escalation

| System | Accuracy | Escalate-F1 | False Auto-Handles |
|--------|----------|-------------|-------------------|
| **Our system** (rule-based) | TBD | TBD | TBD |
| Keyword-only | TBD | TBD | TBD |
| Always escalate | TBD | TBD | 0 |
| Always auto-handle | TBD | TBD | TBD |

### Reply Quality (LLM Judge, 1–5 scale)

| System | Groundedness | Tone | Completeness |
|--------|-------------|------|-------------|
| **Our system** (retrieval-grounded) | TBD | TBD | TBD |
| Zero-shot LLM (no retrieval) | TBD | TBD | TBD |
| Canned reply | TBD | TBD | TBD |

---

## 3. Top 5 Failure Modes

*[To be filled after Phase 7 failure analysis with real examples from eval_examples.csv]*

### Failure Mode 1: TBD
**Examples**: TBD
**Hypothesis**: TBD

### Failure Mode 2: TBD
### Failure Mode 3: TBD
### Failure Mode 4: TBD
### Failure Mode 5: TBD

---

<!-- DRAFT: rewrite in your own words before submitting — you'll be asked to defend this live -->
## 4. What's Misleading About the Headline Number

*[To be filled after Phase 8 with actual numbers]*

Several factors make our headline numbers look better (or worse) than they would in a real deployment:

1. **Class imbalance**: If one intent dominates, accuracy overstates performance. Macro-F1 is more honest, but even macro-F1 gives equal weight to rare intents that might be most important (e.g., billing disputes).

2. **Resolved-thread selection bias**: Our retrieval corpus and golden set are drawn from threads where the brand actually replied. This systematically excludes messages the brand ignored (possibly the hardest or most abusive ones). Our system has never seen these during retrieval or evaluation.

3. **LLM judge self-preference bias**: If the judge model is from the same family as the reply generator, it may rate our replies higher simply because they match its own style. Our judge-vs-human agreement numbers quantify this gap.

4. **Small subsample ≠ full distribution**: We work with ~3K–5K tweets from one brand. The full dataset has ~3M tweets across many brands. Our performance may not generalize to other brands, time periods, or the long tail of rare issues.

5. **Golden set overlap with retrieval corpus**: Some customer messages in the golden set may be semantically similar to messages in the retrieval corpus (since both come from the same brand). This inflates retrieval performance compared to truly novel messages.

---

<!-- DRAFT: rewrite in your own words before submitting — you'll be asked to defend this live -->
## 5. What's Next With One More Week

*[To be filled after Phase 8]*

1. **Fine-tune a classifier**: With the golden set labels, train a small model (e.g., DistilBERT) for intent classification — likely beats few-shot on this narrow domain.
2. **Expand the golden set**: Label 500+ examples for statistical significance and per-class confidence intervals.
3. **Retrieval reranking**: Add a cross-encoder reranker on top of the bi-encoder retrieval for better precedent selection.
4. **Multi-turn context**: Incorporate thread history into the pipeline so the agent can handle follow-up messages.
5. **Better PII protection**: Integrate Presidio or a spaCy NER model for robust PII detection.

---

## 6. Golden Set & Evaluation Methodology

### Sampling
*[To be filled after Phase 5]*

### Labeling
*[To be filled after Phase 5 — describe the labeling process and any difficulties]*

### Judge-Human Agreement
*[To be filled after Phase 6]*
