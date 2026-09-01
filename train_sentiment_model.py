"""
train_sentiment_model.py

Scores each synthetic support ticket with a pretrained sentiment model
(no fine-tuning - distilbert-base-uncased-finetuned-sst-2-english is
used directly), classifies each ticket into a complaint category with
keyword rules, then aggregates to one row per customer.
"""

import os
# Force fully offline/local loading - this environment intermittently
# hits DNS failures on HuggingFace Hub's "is there a newer config"
# connectivity check, which (without these flags) retries with a long
# exponential backoff even though the model is already fully cached
# locally. local_files_only=True below (belt-and-suspenders with these
# env vars) skips that check entirely.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Cap PyTorch's intra-op thread pool - on this environment (Python 3.14
# + torch 2.10 on Windows) leaving it at the auto-detected default adds
# thread-pool overhead disproportionate to a workload this small.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import torch
torch.set_num_threads(1)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tickets_df = pd.read_csv("support_tickets.csv")

# ---------------------------------------------------------------
# 1-2. Pretrained sentiment model, called directly (not through
# transformers' high-level pipeline() wrapper). On this environment,
# pipeline()'s internal batching/iteration hung indefinitely even on
# a handful of short texts, while calling the tokenizer and model
# directly is fast and reliable (verified: 18 texts in ~5s vs. the
# pipeline() call never returning) - so this bypasses that wrapper
# entirely and does the tokenize -> forward -> softmax steps by hand.
# ---------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
print(f"Loading pretrained sentiment model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
sentiment_model.eval()


def score_text(text):
    """One ticket string -> (label, confidence)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_id = probs.argmax().item()
    label = sentiment_model.config.id2label[label_id]
    confidence = probs[label_id].item()
    return label, confidence


# The synthetic ticket text is drawn from a small, fixed set of
# templates (see generate_synthetic_data.py), so most of the ~14K
# rows are exact duplicates of a handful of strings. Scoring each
# UNIQUE text once and mapping the result back to every row is
# mathematically identical to scoring every row individually (the
# model is deterministic for the same input) but turns thousands of
# inference calls into a couple dozen.
unique_texts = tickets_df["ticket_text"].unique().tolist()
print(f"Scoring {len(unique_texts)} unique ticket texts (out of {len(tickets_df)} total tickets)...")

text_to_label = {}
text_to_confidence = {}
for text in unique_texts:
    label, confidence = score_text(text)
    text_to_label[text] = label
    text_to_confidence[text] = confidence

tickets_df["sentiment_label"] = tickets_df["ticket_text"].map(text_to_label)
tickets_df["sentiment_confidence"] = tickets_df["ticket_text"].map(text_to_confidence)
# Convert to a single signed score in [-1, 1]:
# POSITIVE confidence 0.9 -> +0.9, NEGATIVE confidence 0.9 -> -0.9
tickets_df["sentiment_score"] = tickets_df.apply(
    lambda row: row["sentiment_confidence"] if row["sentiment_label"] == "POSITIVE" else -row["sentiment_confidence"],
    axis=1,
)

# ---------------------------------------------------------------
# 3. Complaint category via keyword rules, falling back to sentiment
# for ambiguous cases (no keyword match).
# ---------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Billing": ["bill", "invoice", "charge", "overcharg", "payment", "refund"],
    "Technical": ["internet", "connection", "drop", "wifi", "outage", "device", "signal"],
    "Service Quality": ["technician", "appointment", "hold", "support", "agent", "response", "wait"],
}


def classify_category(text, sentiment_score):
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    # Fallback for ambiguous text with no keyword match: lean on
    # sentiment - a clearly negative ticket with no obvious keyword is
    # bucketed as a general "Service Quality" concern, otherwise "Other".
    if sentiment_score < -0.5:
        return "Service Quality"
    return "Other"


tickets_df["complaint_category"] = tickets_df.apply(
    lambda row: classify_category(row["ticket_text"], row["sentiment_score"]), axis=1
)

# ---------------------------------------------------------------
# 4. Aggregate to one row per customer
# ---------------------------------------------------------------
agg = tickets_df.groupby("customerID").agg(
    sentiment_score=("sentiment_score", "mean"),
    complaint_category=("complaint_category", lambda s: s.mode().iloc[0]),
).reset_index()

agg.to_csv("customer_sentiment_scores.csv", index=False)

print(f"\nScored {len(tickets_df)} tickets across {agg['customerID'].nunique()} customers.")
print("\nSample tickets with computed sentiment:")
print(tickets_df[["customerID", "ticket_text", "sentiment_label", "sentiment_score", "complaint_category"]].head(8).to_string(index=False))
print("\nSample aggregated per-customer scores:")
print(agg.head(5).to_string(index=False))
print("\nSaved customer_sentiment_scores.csv")
