"""
eval/build_eval_set.py

Purpose: build a small "ground truth" set to check whether VADER or the
transformer model is more accurate on YOUR actual chat data.

Since hand-labeling 150 messages is slow, we use "weak labeling":
  1. Randomly sample ~120 messages from your real chat.
  2. Ask a local LLM (via Ollama) to label each one as pos/neg/neutral.
     This is fast but not perfectly reliable on its own.
  3. Export to a CSV. YOU open it and review only a random ~30 of them,
     correcting any label the LLM got wrong. This gives you a
     human-verified subset without labeling all 120 by hand.

Run this ONCE per chat you want to evaluate on. It writes:
  eval/weak_labels.csv        <- full LLM-labeled sample
  eval/review_subset.csv      <- the 30 you should manually check

Usage:
    python eval/build_eval_set.py path/to/whatsapp_chat.txt
"""

import sys
import ollama
import pandas as pd

sys.path.append("..")
from parsers.whatsapp_parser import WhatsAppParser

SAMPLE_SIZE = 120
REVIEW_SIZE = 30

PROMPT_TEMPLATE = """Classify the sentiment of this chat message as exactly
one word: positive, negative, or neutral. Only respond with that one word.

Message: "{message}"
"""


def weak_label_message(message: str) -> str:
    response = ollama.generate(
        model="llama3.2:3b",
        prompt=PROMPT_TEMPLATE.format(message=message),
    )
    text = response["response"].strip().lower()
    for label in ["positive", "negative", "neutral"]:
        if label in text:
            return label
    return "neutral"  # fallback if the model gives an unexpected response


def build_eval_set(chat_file_path: str):
    with open(chat_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    df = WhatsAppParser().parse(raw_text)

    # Drop empty/media-only messages before sampling - they're not useful
    # for a sentiment evaluation set
    usable = df[df["message"].str.strip().str.len() > 3]
    sample = usable.sample(n=min(SAMPLE_SIZE, len(usable)), random_state=42).copy()

    print(f"Weak-labeling {len(sample)} messages using local LLM...")
    sample["weak_label"] = sample["message"].apply(weak_label_message)
    sample["human_verified_label"] = ""  # you'll fill this in for the review subset

    sample[["date", "user", "message", "weak_label", "human_verified_label"]].to_csv(
        "eval/weak_labels.csv", index=False
    )
    print("Saved full weak-labeled set to eval/weak_labels.csv")

    review_subset = sample.sample(n=min(REVIEW_SIZE, len(sample)), random_state=1)
    review_subset[["date", "user", "message", "weak_label", "human_verified_label"]].to_csv(
        "eval/review_subset.csv", index=False
    )
    print(f"Saved {len(review_subset)}-row review subset to eval/review_subset.csv")
    print("\nNext step: open eval/review_subset.csv and fill in 'human_verified_label'")
    print("for each row (positive/negative/neutral) based on your own judgment.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval/build_eval_set.py path/to/whatsapp_chat.txt")
        sys.exit(1)
    build_eval_set(sys.argv[1])
