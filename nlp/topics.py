"""
nlp/topics.py

Topic discovery, beginner-friendly version:

1. Turn each message into a vector ("embedding") using a local
   sentence-transformer model. Similar-meaning messages end up with
   similar vectors - this is what lets us group by MEANING, not just
   shared words.

2. Group the vectors into clusters using KMeans. You pick how many
   clusters (topics) to look for - a reasonable default is 6-8 for a
   typical group chat.

3. For each cluster, run TF-IDF on just that cluster's messages and
   take the top 5 words. That becomes the topic's label
   (e.g. "goa, flight, hotel, booking, trip" -> shown as "Topic 1").

Embeddings are cached to disk (cache/embeddings_cache.npy) so re-running
the app on the same chat doesn't recompute them every time - this was
one of the "cheap but signals engineering awareness" additions we
planned.

Later upgrade path (not done now, kept simple on purpose):
swap KMeans for HDBSCAN so the number of topics is discovered
automatically instead of guessed.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_DIR = "cache"
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _cache_path_for(messages: list[str]) -> str:
    """
    Builds a cache filename based on a hash of the messages.
    If the exact same set of messages is embedded again (e.g. same chat
    re-uploaded), we reuse the cached embeddings instead of recomputing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    joined = "||".join(messages)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"embeddings_{digest}.npy")


def embed_messages(messages: list[str]) -> np.ndarray:
    """Returns embeddings for a list of messages, using disk cache when possible."""
    cache_path = _cache_path_for(messages)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    model = _get_embedding_model()
    embeddings = model.encode(messages, show_progress_bar=False)
    np.save(cache_path, embeddings)
    return embeddings


def label_cluster(messages: list[str], top_n: int = 5) -> str:
    """Gives a cluster a human-readable label using its top TF-IDF terms."""
    if len(messages) < 2:
        return "misc"
    try:
        vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(messages)
        avg_scores = tfidf_matrix.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_indices = avg_scores.argsort()[::-1][:top_n]
        top_terms = [terms[i] for i in top_indices]
        return ", ".join(top_terms)
    except ValueError:
        # Happens if messages are too short/empty for TF-IDF to find terms
        return "misc"


def discover_topics(df: pd.DataFrame, n_topics: int = 6) -> pd.DataFrame:
    """
    Takes the chat dataframe, adds a 'topic_id' and 'topic_label' column,
    and returns a small summary dataframe of topics.

    n_topics: how many topic groups to look for. 6-8 works well for most
    group chats; fewer for smaller/quieter chats.
    """
    usable = df[df["message"].str.strip().str.len() > 3].copy()
    if len(usable) < n_topics:
        n_topics = max(1, len(usable) // 2)

    messages = usable["message"].tolist()
    embeddings = embed_messages(messages)

    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    usable["topic_id"] = kmeans.fit_predict(embeddings)

    topic_summaries = []
    for topic_id in sorted(usable["topic_id"].unique()):
        topic_messages = usable[usable["topic_id"] == topic_id]["message"].tolist()
        label = label_cluster(topic_messages)
        topic_summaries.append(
            {"topic_id": topic_id, "label": label, "message_count": len(topic_messages)}
        )

    usable["topic_label"] = usable["topic_id"].map(
        {t["topic_id"]: t["label"] for t in topic_summaries}
    )

    summary_df = pd.DataFrame(topic_summaries).sort_values(
        "message_count", ascending=False
    )
    return usable, summary_df
