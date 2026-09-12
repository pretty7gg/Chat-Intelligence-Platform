# """
# rag/index.py

# Builds the retrieval half of "Ask your chat" (Retrieval-Augmented Generation).

# Steps:
# 1. Chunk: group messages into small overlapping windows (e.g. 5 messages
#    per chunk) instead of embedding single messages. This gives the LLM
#    enough surrounding context to answer questions properly - a single
#    message like "yes let's do it" means nothing without what came before.

# 2. Embed: turn each chunk into a vector using the same local
#    sentence-transformer model used for topic discovery.

# 3. Store: put those vectors into a FAISS index - a local, free,
#    in-memory search structure that can quickly find the chunks most
#    similar to a user's question.

# Nothing here calls any paid API. FAISS runs entirely on your machine.
# """

# import faiss
# import numpy as np
# import pandas as pd

# from nlp.topics import _get_embedding_model  # reuse the same embedding model


# def chunk_messages(df: pd.DataFrame, chunk_size: int = 5, overlap: int = 2) -> list[dict]:
#     """
#     Groups consecutive messages into overlapping chunks.

#     overlap=2 means each chunk shares its last 2 messages with the next
#     chunk, so a topic that spans a chunk boundary doesn't get cut off
#     awkwardly.

#     Returns a list of dicts, each with the combined text and metadata
#     (date range, senders involved) needed later for citations.
#     """
#     chunks = []
#     step = max(1, chunk_size - overlap)
#     rows = df.to_dict("records")

#     for start in range(0, len(rows), step):
#         window = rows[start:start + chunk_size]
#         if not window:
#             continue

#         text = "\n".join(f"{r['user']}: {r['message']}" for r in window)
#         chunks.append({
#             "text": text,
#             "start_date": window[0]["date"],
#             "end_date": window[-1]["date"],
#             "users": sorted(set(r["user"] for r in window)),
#             "raw_messages": window,  # kept for showing citations later
#         })

#     return chunks


# def build_index(chunks: list[dict]):
#     """
#     Embeds all chunks and builds a FAISS index over them.
#     Returns (index, chunks) - you need both: the index for searching,
#     and the original chunks list to look up what was matched.
#     """
#     model = _get_embedding_model()
#     texts = [c["text"] for c in chunks]
#     embeddings = model.encode(texts, show_progress_bar=False)
#     embeddings = np.array(embeddings).astype("float32")

#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)

#     return index, embeddings


# def search(query: str, index, chunks: list[dict], top_k: int = 3) -> list[dict]:
#     """
#     Given a user's question, finds the top_k most relevant chunks.
#     """
#     model = _get_embedding_model()
#     query_vector = model.encode([query]).astype("float32")

#     distances, indices = index.search(query_vector, top_k)
#     results = []
#     for idx in indices[0]:
#         if 0 <= idx < len(chunks):
#             results.append(chunks[idx])
#     return results


"""
rag/index.py

Retrieval layer for Conversation AI.

Responsibilities:
1. Group consecutive chat messages into overlapping chunks.
2. Embed chunks using the same local sentence-transformer model
   already used by topic discovery.
3. Store embeddings in a local FAISS index.
4. Retrieve relevant conversation chunks for a user question.

Everything runs locally.
"""

import faiss
import numpy as np
import pandas as pd

from nlp.topics import _get_embedding_model


def chunk_messages(
    df: pd.DataFrame,
    chunk_size: int = 5,
    overlap: int = 2
) -> list[dict]:
    """
    Groups consecutive messages into overlapping chunks.

    Each chunk contains:
      - combined conversation text
      - start/end dates
      - users involved
      - raw messages for evidence/citations
    """

    chunks = []

    step = max(1, chunk_size - overlap)
    rows = df.to_dict("records")

    for start in range(0, len(rows), step):
        window = rows[start:start + chunk_size]

        if not window:
            continue

        text = "\n".join(
            f"{r['user']}: {r['message']}"
            for r in window
        )

        chunks.append({
            "text": text,
            "start_date": window[0]["date"],
            "end_date": window[-1]["date"],
            "users": sorted(
                set(str(r["user"]) for r in window)
            ),
            "raw_messages": window,
        })

    return chunks


def build_index(chunks: list[dict]):
    """
    Creates a FAISS vector index from conversation chunks.

    Returns:
        index
        embeddings
    """

    if not chunks:
        raise ValueError("No conversation chunks available.")

    model = _get_embedding_model()

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def search(
    query: str,
    index,
    chunks: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Retrieves the most semantically relevant chunks.

    A slightly larger candidate pool is searched and then reduced
    to top_k so that exact conversation terms are more likely to
    be represented.
    """

    if index is None or not chunks:
        return []

    model = _get_embedding_model()

    query_vector = model.encode(
        [query],
        show_progress_bar=False
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    )

    candidate_k = min(
        max(top_k * 3, 10),
        len(chunks)
    )

    distances, indices = index.search(
        query_vector,
        candidate_k
    )

    candidates = []

    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx].copy()

            # Lower distance = more semantically similar.
            chunk["_distance"] = float(distances[0][rank])
            chunk["_semantic_rank"] = rank

            candidates.append(chunk)

    # Normalize semantic scores.
    if candidates:
        max_distance = max(
            c["_distance"] for c in candidates
        )

        min_distance = min(
            c["_distance"] for c in candidates
        )

        distance_range = max_distance - min_distance

        for c in candidates:
            if distance_range > 0:
                c["_semantic_score"] = (
                    max_distance - c["_distance"]
                ) / distance_range
            else:
                c["_semantic_score"] = 1.0

    # Give exact query terms a small boost.
    query_terms = {
        word.lower()
        for word in query.split()
        if len(word) > 2
    }

    for c in candidates:
        text_lower = c["text"].lower()

        exact_matches = sum(
            1
            for term in query_terms
            if term in text_lower
        )

        keyword_score = (
            exact_matches / len(query_terms)
            if query_terms
            else 0
        )

        c["_final_score"] = (
            0.80 * c["_semantic_score"]
            + 0.20 * keyword_score
        )

    candidates.sort(
        key=lambda x: x["_final_score"],
        reverse=True
    )

    # Remove internal retrieval fields.
    results = []

    for chunk in candidates[:top_k]:
        cleaned = {
            key: value
            for key, value in chunk.items()
            if not key.startswith("_")
        }
        results.append(cleaned)

    return results