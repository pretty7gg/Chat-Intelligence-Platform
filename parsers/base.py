"""
base.py

This file defines the "contract" that every chat parser must follow.

Why this exists:
Right now we only support WhatsApp exports. But WhatsApp, Telegram, Discord,
and Slack exports all look different as raw text/JSON. Instead of writing
one giant messy function that tries to handle all of them, we define ONE
common output format (a pandas DataFrame with fixed column names). Every
parser's only job is: take a platform's raw export -> produce a DataFrame
with these exact columns.

This means the rest of the app (stats, sentiment, topics, RAG) never needs
to know which platform the data came from. That's the "generalize beyond
WhatsApp" upgrade from our plan, done in a way that doesn't require
rewriting anything else later.
"""

import pandas as pd

# Every parser MUST return a DataFrame with exactly these columns.
REQUIRED_COLUMNS = ["date", "user", "message"]


class BaseChatParser:
    """
    Abstract base class. Every specific parser (WhatsAppParser, etc.)
    should inherit from this and implement `parse`.
    """

    def parse(self, raw_text: str) -> pd.DataFrame:
        """
        Takes raw exported chat text and returns a standardized DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement parse()")

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that a parser's output actually matches our required schema.
        Call this at the end of every parser's parse() method as a safety net.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Parser output is missing required columns: {missing}. "
                f"Every parser must return columns: {REQUIRED_COLUMNS}"
            )
        return df
