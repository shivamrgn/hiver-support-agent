# Hiver SDE Intern — AI Customer Support Agent

An AI-powered customer support agent built for **[brand TBD after Phase 1 data analysis]** using the [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

The agent:
1. **Classifies** incoming customer messages into intent categories
2. **Drafts replies** grounded in how the brand has historically resolved similar issues
3. **Decides** auto-handle vs. escalate-to-human, with a stated reason every time

## Quick Start

### Prerequisites
- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, or Google Gemini)

### Setup
```bash
# 1. Create virtualenv and install dependencies
make setup

# 2. Activate the environment
source .venv/bin/activate

# 3. Configure API key(s)
cp .env.example .env
# Edit .env to add your API key(s)
```

### Reproduce Headline Numbers (<15 min)
```bash
# Uses cached model outputs — no API calls needed
make reproduce
```

This reads from `results/cache/` and recomputes all metrics from cached outputs.

### Fresh Evaluation (requires API key)
```bash
# Calls LLM APIs to regenerate all outputs from scratch
make eval-fresh
```

### Run Tests
```bash
make test
```

## Project Structure

```
├── src/                    # Core agent modules
│   ├── pipeline.py         # End-to-end orchestrator
│   ├── intent_classifier.py # LLM few-shot classification
│   ├── retrieval.py        # Embedding index + top-k retrieval
│   ├── reply_generator.py  # Retrieval-grounded reply drafting
│   ├── escalation.py       # Interpretable rule/threshold logic
│   ├── baselines.py        # Trivial + simple baselines
│   ├── llm_client.py       # Multi-provider LLM wrapper
│   └── pii_scrubber.py     # PII removal from precedents
├── eval/                   # Evaluation harness
│   ├── run_eval.py         # Single entrypoint
│   ├── metrics.py          # Classification + groundedness metrics
│   ├── llm_judge.py        # LLM-as-judge scoring
│   ├── llm_judge_rubric.md # Versioned scoring rubric
│   └── judge_human_agreement.py
├── scripts/                # Data prep + labeling
│   ├── sample_golden_set.py
│   └── label_golden_set.py # Streamlit labeling UI
├── data/                   # Checked-in subsample + golden set
├── results/                # Metrics, examples, cache
└── report/                 # Report, decision log, citations
```

## Architecture

```
Customer Message
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   Intent     │     │  Retrieval   │
│  Classifier  │     │    Index     │
│  (LLM few-   │     │ (sentence-   │
│   shot)      │     │ transformers)│
└──────┬───────┘     └──────┬───────┘
       │                    │
       │    ┌───────────────┘
       │    │ top-k precedents
       │    │ (PII-scrubbed)
       ▼    ▼
┌──────────────────┐
│  Reply Generator │
│  (LLM, grounded  │
│  in precedents)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Escalation     │
│ (rule/threshold  │
│  logic: intent   │
│  risk + conf +   │
│  similarity +    │
│  keywords)       │
└────────┬─────────┘
         │
         ▼
{intent, reply_draft, decision, reason}
```

## Data & Methodology

- **Dataset**: Customer Support on Twitter (~3M tweets, Thought Vector, CC BY-NC-SA 4.0)
- **Subsample**: ~3K–5K tweets for the chosen brand (checked into repo)
- **Retrieval corpus**: Resolved customer→company pairs with PII scrubbing
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (local, free)
- **Golden set**: 150–250 hand-labeled examples, stratified sampling

## Key Design Decisions

See [report/DECISION_LOG.md](report/DECISION_LOG.md) for the full log.

## Report

See [report/REPORT.md](report/REPORT.md) for the full evaluation report, including:
- Problem framing and scope cuts
- Results vs. baselines
- Top 5 failure modes with real examples
- What's misleading about the headline numbers
- What's next with one more week

## Citations

See [report/CITATIONS.md](report/CITATIONS.md).

## License

This project was built as a take-home assignment submission. The dataset is licensed under CC BY-NC-SA 4.0.
