# LLM-as-Judge Rubric (v1.0)

This rubric is used by `eval/llm_judge.py` to score the quality of drafted
customer support replies. It is versioned and kept human-readable so it can
be inspected, modified, and defended in review.

## Scoring Dimensions

Each dimension is scored on a 1–5 scale.

---

### 1. Groundedness (Is the reply supported by the retrieved precedent?)

| Score | Criteria |
|-------|----------|
| 5 | Reply clearly draws from the precedent's resolution pattern; specific steps or information from the precedent are paraphrased (not copied). |
| 4 | Reply is mostly grounded in the precedent but adds minor details not supported by it. |
| 3 | Reply references the general area of the precedent but is vague or generic. |
| 2 | Reply is loosely related to the precedent but mostly generates its own content. |
| 1 | Reply ignores the precedent entirely or contradicts it. |

**Anchor example (5):** Precedent says "clear cache in Settings > Storage." Reply says "Try going to your Settings, then Storage, and clearing the cache — that often fixes this."

**Anchor example (1):** Precedent says "clear cache in Settings > Storage." Reply says "Have you tried reinstalling the app?"

---

### 2. Tone & Brand-Voice Fit (Does it sound like the brand's actual support account?)

| Score | Criteria |
|-------|----------|
| 5 | Tone perfectly matches the brand's support style: empathetic but not over-the-top, concise, professional, uses appropriate personalization. |
| 4 | Tone is good but slightly too formal/informal or generic compared to how the brand actually writes. |
| 3 | Tone is acceptable but reads like a generic chatbot, not this specific brand. |
| 2 | Tone is off — too casual, too stiff, or doesn't match the support context. |
| 1 | Tone is inappropriate — dismissive, robotic, or unprofessional. |

**Anchor example (5):** "Hey! Sorry about that 😊 Let's get this sorted out. Can you DM us your account email so we can take a closer look?"

**Anchor example (1):** "Your request has been logged. Please await further communication."

---

### 3. Completeness (Does the reply fully address the customer's issue?)

| Score | Criteria |
|-------|----------|
| 5 | Reply addresses every aspect of the customer's message, provides clear next steps or resolution. |
| 4 | Reply addresses the main issue and provides next steps, but misses a minor detail from the customer's message. |
| 3 | Reply addresses the main issue but is vague on next steps. |
| 2 | Reply partially addresses the issue or provides a generic "we'll look into it" without specifics. |
| 1 | Reply doesn't address the customer's actual issue, or is completely off-topic. |

**Anchor example (5):** Customer says "I was charged twice." Reply acknowledges the double charge, apologizes, and directs to DM for a refund check.

**Anchor example (1):** Customer says "I was charged twice." Reply says "Thanks for reaching out! We're here to help."

---

## Scoring Instructions for the Judge

1. Read the customer message carefully.
2. Read the retrieved precedent(s) used for grounding.
3. Read the drafted reply.
4. Score each of the three dimensions independently (1–5).
5. **Penalize generic/canned-sounding replies** even if they're technically correct — the point of retrieval grounding is to produce specific, relevant responses.
6. Do NOT give bonus points for length — concise replies that address the issue are preferred.

## Known Limitation

When the LLM judge is the same model family as the reply drafter, there is a
risk of **self-preference bias** — the judge may rate replies higher simply
because they match its own style. To mitigate this:
- We use a different model or a clearly separated persona for judging.
- We compare judge scores against human scores (judge-human agreement).
- We report this limitation in the final report.
