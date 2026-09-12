"""
helper.py

Same stats/timeline/activity functions as your original project - these
are still useful and we're not rewriting good code for no reason.

REMOVED (as planned):
  - emoji_helper()          -> emoji analysis added no real insight
  - most_common_words()     -> replaced entirely by nlp/topics.py

Sentiment percentage/breakdown functions are now generalized to work
with EITHER 'vader_value' or 'transformer_value' columns (pass the
column name in), since we now have two sentiment sources instead of one.
"""

import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud

extract = URLExtract()


def fetch_stats(selected_user: str, df: pd.DataFrame):
    """Returns (num_messages, num_words, num_media, num_links, num_deleted)."""
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df["message"]:
        words.extend(message.split())

    num_media_messages = df[df["message"] == "<Media omitted>\n"].shape[0]

    links = []
    for message in df["message"]:
        links.extend(extract.find_urls(message))

    num_deleted_messages = df[df["message"] == "This message was deleted\n"].shape[0]

    return num_messages, len(words), num_media_messages, len(links), num_deleted_messages


def most_busy_users(df: pd.DataFrame):
    """Returns (top-5 counts series, percentage-per-user dataframe)."""
    top_counts = df["user"].value_counts().head()
    percent_df = round((df["user"].value_counts() / df.shape[0]) * 100, 2).reset_index()
    percent_df.columns = ["name", "percent"]
    return top_counts, percent_df


def create_wordcloud(selected_user: str, df: pd.DataFrame, stop_words: str):
    """
    General wordcloud across all messages (or one user's messages).
    Sentiment-specific wordclouds were removed - the Topics tab now
    answers "what did people talk about" more meaningfully than word
    clouds ever did.
    """
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["message"] != "<Media omitted>\n"]

    def remove_stop_words(message):
        return " ".join(
            word for word in message.lower().split() if word not in stop_words
        )

    cleaned = temp["message"].apply(remove_stop_words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    return wc.generate(cleaned.str.cat(sep=" "))


def monthly_timeline(selected_user: str, df: pd.DataFrame):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()
    timeline["time"] = timeline["month"] + "-" + timeline["year"].astype(str)
    return timeline


def week_activity_map(selected_user: str, df: pd.DataFrame):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    return df["day_name"].value_counts()


def month_activity_map(selected_user: str, df: pd.DataFrame):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    return df["month"].value_counts()


def activity_heatmap(selected_user: str, df: pd.DataFrame):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    return df.pivot_table(
        index="day_name", columns="period", values="message", aggfunc="count"
    ).fillna(0)


def sentiment_percentage(df: pd.DataFrame, sentiment_column: str, value: int):
    """
    Generalized version of your original percentage() function.
    sentiment_column: 'vader_value' or 'transformer_value'
    value: -1 (negative), 0 (neutral), or 1 (positive)
    """
    subset = df[df[sentiment_column] == value]
    if subset.empty:
        return pd.DataFrame(columns=["name", "percent"])

    percent_df = round(
        (subset["user"].value_counts() / subset.shape[0]) * 100, 2
    ).reset_index()
    percent_df.columns = ["name", "percent"]
    return percent_df
