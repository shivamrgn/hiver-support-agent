# Decision Log

Every non-obvious decision is logged here at the moment it's made, not reconstructed after the fact. Each entry includes the decision, the alternatives considered, and the reason for the choice.

---

1. **Brand selection: TBD after Phase 1 data analysis.** Will compare top brands on: inbound volume, % resolved threads, English-language share, and average thread length. Prefer a brand with 5K–20K inbound tweets (enough for retrieval corpus, not so many that the subsample is hard to manage), high resolved-thread rate, and a focused problem domain.

2. **Embedding model: `all-MiniLM-L6-v2`** over larger models (`all-mpnet-base-v2`, OpenAI `text-embedding-3-small`). Reason: runs locally (no API cost, no key required for retrieval index), 384-dim embeddings are sufficient for cosine similarity over a few thousand vectors, and it's the most commonly-benchmarked small model. Trade-off: lower absolute quality than larger models, but retrieval is over a small, domain-specific corpus where even moderate embeddings perform well.

3. **Vector index: plain numpy cosine similarity** over FAISS/ChromaDB. Reason: with ~2K–5K resolved pairs, brute-force cosine similarity runs in <100ms. Adding FAISS would add a dependency and abstraction layer for zero measurable benefit at this scale. Logged because a reviewer might ask "why not FAISS?" — answer: the corpus is too small to need it.

4. **LLM client: multi-provider wrapper** (OpenAI / Anthropic / Google) with auto-detection. Reason: the assignment allows any LLM API, and hardcoding one vendor is unnecessary coupling. The wrapper is thin (~60 lines) and makes the project portable.

5. **Intent classification approach: LLM few-shot** over fine-tuned classifier. Reason: with a small subsample and no pre-labeled training set, few-shot classification avoids the cold-start problem. A fine-tuned model would need labeled data we don't have yet (circular with golden set). TF-IDF + logistic regression serves as the simple baseline to compare against.

6. **Labeling UI: Streamlit** over terminal loop or Gradio. Reason: fastest to build with decent UX (dropdowns, progress bar, keyboard-friendly), familiar to most Python developers, and requires no frontend knowledge. Terminal loop would be faster to code but slower and more error-prone for the human labeler.

7. **"Resolved" thread heuristic**: A customer→company pair is considered "resolved" if: (a) the company replied, AND (b) the thread either ends after the company reply or the customer's follow-up is a "thank you" / positive sentiment (not a repeated complaint). This is a best-effort heuristic — some "resolved" pairs may actually be unresolved. Logged as a known limitation.

8. **PII scrubbing approach: regex-based** over NER-based (spaCy/Presidio). Reason: the dataset already masks `__email__` and some phone numbers, so we primarily need to catch residual patterns (unmmasked @mentions, "my name is X", order numbers). Regex is simpler, faster, and sufficient for the patterns we expect. A production system would use Presidio. Logged as a scope cut.

9. **AI-assisted development**: This entire project was built with the assistance of an AI coding agent (Google Antigravity / Claude). All significant code was reviewed and understood by the submitter. Logged per assignment rules requiring citation of AI assistance.

10. **Confidence score approach: LLM self-reported** (1–5 scale) over logprob-based. Reason: not all providers expose logprobs, and we need provider-agnostic confidence. The LLM is prompted to assess its own certainty. This is a known weak signal — logged as a limitation and used conservatively (only low confidence triggers escalation, never used to assert high confidence).
