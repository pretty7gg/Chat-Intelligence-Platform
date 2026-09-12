# """
# nlp/sentiment.py

# Two sentiment approaches, kept side by side on purpose:

# 1. VADER (lexicon/rule-based) - your original approach. Fast, no download
#    needed, but struggles with sarcasm, negation, and context.

# 2. A local transformer model (cardiffnlp/twitter-roberta-base-sentiment-latest)
#    - understands context/negation better, but is heavier and slower.
#    Runs fully offline once downloaded - no paid API calls.

# We don't throw VADER away. We keep both so the app can show where they
# agree/disagree, which is the actual interesting finding for your project.
# """

# import nltk
# import pandas as pd
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download("vader_lexicon", quiet=True)

# _vader = SentimentIntensityAnalyzer()

# # Transformer model is loaded lazily (only when first needed) because
# # it's slow to load and we don't want to pay that cost if the user
# # never opens the sentiment tab.
# _transformer_pipeline = None


# def _get_transformer_pipeline():
#     global _transformer_pipeline
#     if _transformer_pipeline is None:
#         from transformers import pipeline
#         _transformer_pipeline = pipeline(
#             "sentiment-analysis",
#             model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#         )
#     return _transformer_pipeline


# def vader_label(message: str) -> int:
#     """
#     Returns -1 (negative), 0 (neutral), or 1 (positive) using VADER.
#     This is your original logic, unchanged.
#     """
#     scores = _vader.polarity_scores(message)
#     pos, neg, neu = scores["pos"], scores["neg"], scores["neu"]
#     if pos >= neg and pos >= neu:
#         return 1
#     if neg >= pos and neg >= neu:
#         return -1
#     return 0


# def transformer_label(message: str) -> int:
#     """
#     Returns -1, 0, or 1 using the local transformer model.
#     The model's raw labels are 'negative' / 'neutral' / 'positive'.
#     """
#     if not message.strip():
#         return 0
#     clf = _get_transformer_pipeline()
#     # Transformer models have a max input length - truncate very long messages
#     result = clf(message[:512])[0]
#     label = result["label"].lower()
#     mapping = {"negative": -1, "neutral": 0, "positive": 1}
#     return mapping.get(label, 0)


# def add_sentiment_columns(df: pd.DataFrame, use_transformer: bool = True) -> pd.DataFrame:
#     """
#     Adds two columns to the dataframe:
#       - 'vader_value': -1/0/1 from VADER
#       - 'transformer_value': -1/0/1 from the transformer (if enabled)

#     use_transformer=False lets you skip the slower model, e.g. for quick testing.
#     """
#     df = df.copy()
#     df["vader_value"] = df["message"].apply(vader_label)

#     if use_transformer:
#         df["transformer_value"] = df["message"].apply(transformer_label)
#     else:
#         df["transformer_value"] = None

#     return df


"""
nlp/sentiment.py

Two sentiment approaches, kept side by side on purpose:

1. VADER (lexicon/rule-based) - your original approach. Fast, no download
   needed, but struggles with sarcasm, negation, and context.

2. A local transformer model (cardiffnlp/twitter-roberta-base-sentiment-latest)
   - understands context/negation better, but is heavier and slower.
   Runs fully offline once downloaded - no paid API calls.

We don't throw VADER away. We keep both so the app can show where they
agree/disagree, which is the actual interesting finding for your project.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()

# Transformer model is loaded lazily (only when first needed) because
# it's slow to load and we don't want to pay that cost if the user
# never opens the sentiment tab.
_transformer_pipeline = None


def _get_transformer_pipeline():
    global _transformer_pipeline
    if _transformer_pipeline is None:
        import torch
        torch.set_num_threads(1)
        from transformers import pipeline
        _transformer_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,  # force CPU explicitly, avoids ambiguous device detection
        )
    return _transformer_pipeline


def vader_label(message: str) -> int:
    """
    Returns -1 (negative), 0 (neutral), or 1 (positive) using VADER.
    This is your original logic, unchanged.
    """
    scores = _vader.polarity_scores(message)
    pos, neg, neu = scores["pos"], scores["neg"], scores["neu"]
    if pos >= neg and pos >= neu:
        return 1
    if neg >= pos and neg >= neu:
        return -1
    return 0


def transformer_label(message: str) -> int:
    """
    Returns -1, 0, or 1 using the local transformer model.
    The model's raw labels are 'negative' / 'neutral' / 'positive'.
    """
    if not message.strip():
        return 0
    clf = _get_transformer_pipeline()
    # Transformer models have a max input length - truncate very long messages
    result = clf(message[:512])[0]
    label = result["label"].lower()
    mapping = {"negative": -1, "neutral": 0, "positive": 1}
    return mapping.get(label, 0)


def add_sentiment_columns(df: pd.DataFrame, use_transformer: bool = True) -> pd.DataFrame:
    """
    Adds two columns to the dataframe:
      - 'vader_value': -1/0/1 from VADER
      - 'transformer_value': -1/0/1 from the transformer (if enabled)

    use_transformer=False lets you skip the slower model, e.g. for quick testing.
    """
    df = df.copy()
    df["vader_value"] = df["message"].apply(vader_label)

    if use_transformer:
        df["transformer_value"] = df["message"].apply(transformer_label)
    else:
        df["transformer_value"] = None

    return df