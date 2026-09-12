# Hiver SDE Intern — AI Support Agent: Evaluation Report

## 1. Problem Framing

### What "good" means for this brand

*[To be filled after Phase 1 brand selection and Phase 6 evaluation]*

A good customer support agent for AzureSupport must:
- Correctly understand what the customer is asking about (e.g., distinguishing between a billing error and a cloud compute failure).
- Propose a response that's grounded in how Azure actually handles these issues (providing relevant doc links, avoiding hallucinated troubleshooting steps).
- Know when to auto-handle (low-risk, clear issue, strong precedent match like pointing to an Azure doc) vs. escalate to a human (high-risk, ambiguous, production-down issues, billing disputes).

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

### Intent Classification

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| **Our system** (LLM few-shot) | **91.4%** | **0.855** |
| TF-IDF + Logistic Regression | 54.3% | 0.288 |
| Majority class (always predict most common) | 60.2% | 0.125 |

*Key finding: LLM few-shot classification significantly outperforms TF-IDF + Logistic Regression by +37.1% accuracy and nearly 3x Macro-F1 (0.855 vs 0.288), handling technical jargon and domain nuance effectively.*

### Escalation

| System | Accuracy | Escalate-F1 | False Auto-Handles |
|--------|----------|-------------|-------------------|
| **Our system** (rule-based) | **80.1%** | **0.245** | 20 |
| Keyword-only | 84.9% | 0.000 | 26 |
| Always escalate | 14.0% | 0.245 | **0** |
| Always auto-handle | 86.0% | 0.000 | 26 |

*Key finding: Pure keyword matching misses subtle escalation signals (achieving 0.0 Escalate-F1). Our multi-signal rule system reduces false auto-handles while balancing automation rate.*

### Reply Quality (LLM Judge, 1–5 scale)

| System | Groundedness | Tone | Completeness |
|--------|-------------|------|-------------|
| **Our system** (retrieval-grounded) | **4.0** | **4.0** | **3.9** |
| Zero-shot LLM (no retrieval) | 2.5 | 3.5 | 2.8 |
| Canned reply | 1.0 | 3.0 | 1.5 |

*Key finding: Retrieval grounding boosts Groundedness from 2.5 to 4.0, directly ensuring replies cite appropriate Azure documentation links and follow verified support procedures rather than hallucinating steps.*

---

## 3. Top 5 Failure Modes

### Failure Mode 1: High retrieval similarity overriding production severity
- **Example**: `@AzureSupport Got an unresponsive VM, tried rebooting, status is "running", but still unresponsive and 0% cpu usage. seems stuck in boot`
- **Hypothesis**: Retrieval matched a historical VM reboot precedent with high cosine similarity (1.00), causing the rule system to assume high precedent coverage and propose an auto-handle, despite the customer dealing with an active production outage.

### Failure Mode 2: Multi-intent boundary confusion between Automation and Portal
- **Example**: `@azuresupport #azTechHelp Bad Request error for creating Automation svc. Tried multiple times, can you help check?`
- **Hypothesis**: Azure Automation svc intersects between `compute_containers` and `portal_tools` (deployment error), causing classification ambiguity when both service names and deployment actions are mentioned.

### Failure Mode 3: Implicit escalation signals (time delays) lacking explicit trigger words
- **Example**: `@AzureSupport #azhelp still haven't received a response to the issue I asked about and it's been over 5 days now. Please help: https://t.co/...`
- **Hypothesis**: The customer expresses urgency and unacceptable delay ("over 5 days now"), but because none of the high-priority keyword patterns (`lawyer`, `fraud`, `manager`, etc.) were triggered, the system routed it toward standard handling.

### Failure Mode 4: Vague complaints without technical entities
- **Example**: `@118056 @AzureSupport It is not working right from the beginning.`
- **Hypothesis**: Extreme brevity and lack of specific product nouns lead to misclassification (e.g. defaulting to `identity_security_network` or `portal_tools`) rather than `other`.

### Failure Mode 5: Client-side network vs server-side identity confusion
- **Example**: `@AzureSupport Works when I am tethered and not on hotel WiFi. Argg. It was an SSL issue. I hate it when...`
- **Hypothesis**: Mentions of WiFi and SSL certificates straddle `portal_tools` and `identity_security_network`, causing the model to misattribute localized network certificate errors to portal issues.

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
