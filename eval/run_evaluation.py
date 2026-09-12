"""
eval/run_evaluation.py

After you've filled in 'human_verified_label' in eval/review_subset.csv,
run this to see how VADER and the transformer model each compare against
your own judgment. This produces the numbers/table you'll put in the README.

Usage:
    python eval/run_evaluation.py
"""

import sys
import pandas as pd

sys.path.append("..")
from nlp.sentiment import vader_label, transformer_label

LABEL_TO_INT = {"positive": 1, "negative": -1, "neutral": 0}
INT_TO_LABEL = {1: "positive", -1: "negative", 0: "neutral"}


def run_evaluation():
    df = pd.read_csv("eval/review_subset.csv")

    df = df[df["human_verified_label"].str.strip() != ""].copy()
    if df.empty:
        print("No rows have 'human_verified_label' filled in yet.")
        print("Open eval/review_subset.csv, fill that column in, then rerun.")
        return

    df["true_int"] = df["human_verified_label"].str.lower().map(LABEL_TO_INT)

    df["vader_pred"] = df["message"].apply(vader_label)
    print("Running transformer model (this may take a moment)...")
    df["transformer_pred"] = df["message"].apply(transformer_label)

    vader_accuracy = (df["vader_pred"] == df["true_int"]).mean()
    transformer_accuracy = (df["transformer_pred"] == df["true_int"]).mean()

    print(f"\nEvaluated on {len(df)} human-verified messages\n")
    print(f"VADER accuracy:        {vader_accuracy:.2%}")
    print(f"Transformer accuracy:  {transformer_accuracy:.2%}\n")

    disagreements = df[df["vader_pred"] != df["transformer_pred"]]
    print(f"Found {len(disagreements)} messages where VADER and transformer disagreed:")
    for _, row in disagreements.head(10).iterrows():
        print(f"\n  Message: {row['message'][:80]}")
        print(f"  Human label:  {row['human_verified_label']}")
        print(f"  VADER said:   {INT_TO_LABEL[row['vader_pred']]}")
        print(f"  Transformer said: {INT_TO_LABEL[row['transformer_pred']]}")

    df.to_csv("eval/evaluation_results.csv", index=False)
    print("\nFull results saved to eval/evaluation_results.csv")


if __name__ == "__main__":
    run_evaluation()
