"""
Golden set labeling tool (Streamlit app).

A lightweight UI for a human labeler to quickly classify customer messages
into intents and decide escalation.

Usage:
    streamlit run scripts/label_golden_set.py
    # or: make label
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from src.intent_classifier import INTENT_TAXONOMY

DATA_DIR = Path(__file__).parent.parent / "data"
UNLABELED_PATH = DATA_DIR / "golden_set_unlabeled.csv"
LABELED_PATH = DATA_DIR / "golden_set_labeled.csv"

# Intent options for dropdown
INTENT_OPTIONS = [""] + [
    f"{key}: {info['label']}"
    for key, info in INTENT_TAXONOMY.items()
] + ["ambiguous: Ambiguous / Multiple Intents"]

ESCALATION_OPTIONS = ["", "auto_handle", "escalate"]


def load_data() -> pd.DataFrame:
    """Load the golden set, preferring partially-labeled file if it exists."""
    if LABELED_PATH.exists():
        return pd.read_csv(LABELED_PATH)
    elif UNLABELED_PATH.exists():
        return pd.read_csv(UNLABELED_PATH)
    else:
        st.error(
            f"No golden set found at {UNLABELED_PATH}. "
            "Run `python scripts/sample_golden_set.py` first."
        )
        st.stop()


def save_data(df: pd.DataFrame):
    """Save progress to the labeled CSV."""
    df.to_csv(LABELED_PATH, index=False)


def main():
    st.set_page_config(page_title="Golden Set Labeler", layout="wide")
    st.title("🏷️ Golden Set Labeler")
    st.markdown("Label customer messages with intent and escalation decision.")

    # Load data
    df = load_data()

    # Ensure label columns exist
    for col in ["intent_label", "escalation_label", "notes"]:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna({"intent_label": "", "escalation_label": "", "notes": ""})

    # Progress tracking
    total = len(df)
    labeled = df[
        (df["intent_label"].astype(str).str.strip() != "") &
        (df["escalation_label"].astype(str).str.strip() != "")
    ].shape[0]

    st.progress(labeled / total if total > 0 else 0)
    st.markdown(f"**Progress: {labeled}/{total}** ({labeled/total*100:.0f}%)")

    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if "current_idx" not in st.session_state:
            # Start at first unlabeled
            unlabeled_idx = df[
                (df["intent_label"].astype(str).str.strip() == "") |
                (df["escalation_label"].astype(str).str.strip() == "")
            ].index
            st.session_state.current_idx = int(unlabeled_idx[0]) if len(unlabeled_idx) > 0 else 0

        idx = st.number_input(
            "Go to example #",
            min_value=0,
            max_value=total - 1,
            value=st.session_state.current_idx,
            key="nav_input",
        )
        st.session_state.current_idx = idx

    with col3:
        jump_next = st.button("⏭️ Next Unlabeled")
        if jump_next:
            unlabeled_idx = df[
                (df["intent_label"].astype(str).str.strip() == "") |
                (df["escalation_label"].astype(str).str.strip() == "")
            ].index
            if len(unlabeled_idx) > 0:
                st.session_state.current_idx = int(unlabeled_idx[0])
                st.rerun()

    # Current example
    row = df.iloc[idx]

    st.divider()

    # Display message
    col_msg, col_meta = st.columns([3, 1])
    with col_msg:
        st.subheader(f"Example {idx + 1} / {total}")
        st.markdown(f"**Customer message:**")
        st.info(row.get("text", "(no text)"))

    with col_meta:
        st.markdown(f"**Tweet ID:** `{row.get('tweet_id', 'N/A')}`")
        st.markdown(f"**Sample bucket:** `{row.get('sample_bucket', 'N/A')}`")
        if "created_at" in row and pd.notna(row["created_at"]):
            st.markdown(f"**Date:** `{row['created_at']}`")

    # Labeling inputs
    st.divider()
    col_intent, col_esc, col_notes = st.columns([2, 1, 2])

    with col_intent:
        current_intent = str(row.get("intent_label", ""))
        intent_idx = 0
        for i, opt in enumerate(INTENT_OPTIONS):
            if opt.startswith(current_intent.split(":")[0].strip()):
                intent_idx = i
                break

        intent = st.selectbox(
            "Intent category",
            INTENT_OPTIONS,
            index=intent_idx,
            key=f"intent_{idx}",
        )

    with col_esc:
        current_esc = str(row.get("escalation_label", ""))
        esc_idx = 0
        if current_esc in ESCALATION_OPTIONS:
            esc_idx = ESCALATION_OPTIONS.index(current_esc)

        escalation = st.radio(
            "Escalation decision",
            ESCALATION_OPTIONS[1:],  # skip empty
            index=max(0, esc_idx - 1) if esc_idx > 0 else 0,
            key=f"esc_{idx}",
        )

    with col_notes:
        notes = st.text_area(
            "Notes (optional)",
            value=str(row.get("notes", "")),
            key=f"notes_{idx}",
        )

    # Save & Navigate buttons
    col_save, col_prev, col_next = st.columns([2, 1, 1])

    with col_save:
        if st.button("💾 Save & Next", type="primary"):
            # Save labels
            intent_key = intent.split(":")[0].strip() if intent else ""
            df.at[idx, "intent_label"] = intent_key
            df.at[idx, "escalation_label"] = escalation
            df.at[idx, "notes"] = notes
            save_data(df)

            # Move to next
            if idx < total - 1:
                st.session_state.current_idx = idx + 1
            st.rerun()

    with col_prev:
        if st.button("◀ Previous") and idx > 0:
            st.session_state.current_idx = idx - 1
            st.rerun()

    with col_next:
        if st.button("Next ▶") and idx < total - 1:
            st.session_state.current_idx = idx + 1
            st.rerun()

    # Taxonomy reference (collapsible)
    with st.expander("📋 Intent Taxonomy Reference"):
        for key, info in INTENT_TAXONOMY.items():
            st.markdown(f"**{key}**: {info['label']}")
            st.markdown(f"  _{info['definition']}_")
            for ex in info["examples"]:
                st.markdown(f"  - {ex}")
            st.markdown("")


if __name__ == "__main__":
    main()
