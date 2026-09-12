# """
# rag/qa.py

# The second half of "Ask your chat": takes a user's question, retrieves
# relevant message chunks (via rag/index.py), and asks a LOCAL LLM (via
# Ollama - free, offline) to answer using ONLY those chunks.

# The prompt explicitly tells the model not to make things up beyond what's
# in the retrieved messages. This is what "grounding" means, and it's why
# we can show citations afterward - the answer is supposed to be traceable
# back to real messages, not invented.
# """

# import ollama

# from rag.index import search

# ANSWER_PROMPT = """You are analyzing a private group chat. Answer the
# question using ONLY the conversation excerpts below. If the excerpts
# don't contain enough information to answer, say so honestly instead of
# guessing.

# Conversation excerpts:
# {context}

# Question: {question}

# Give a short, direct answer (2-3 sentences max).
# """


# def answer_question(question: str, index, chunks: list[dict], top_k: int = 3) -> dict:
#     """
#     Returns a dict with:
#       - 'answer': the LLM's text answer
#       - 'evidence': the raw chunks used, for displaying citations in the UI
#     """
#     relevant_chunks = search(question, index, chunks, top_k=top_k)

#     context = "\n\n---\n\n".join(c["text"] for c in relevant_chunks)
#     prompt = ANSWER_PROMPT.format(context=context, question=question)

#     response = ollama.generate(model="llama3.1:8b", prompt=prompt)
#     answer_text = response["response"].strip()

#     return {
#         "answer": answer_text,
#         "evidence": relevant_chunks,
#     }


# def format_evidence_for_display(evidence: list[dict]) -> str:
#     """
#     Turns the raw evidence chunks into the citation format we planned:

#     12 Aug 2025 - Pretty: Let's plan the Goa trip...
#     14 Aug 2025 - Rahul: Goa hotel booking?
#     """
#     lines = []
#     for chunk in evidence:
#         for msg in chunk["raw_messages"]:
#             date_str = msg["date"].strftime("%d %b %Y")
#             snippet = msg["message"].strip()[:80]
#             lines.append(f"{date_str} - {msg['user']}: {snippet}")
#     return "\n".join(lines)


"""
rag/qa.py

Conversation AI layer.

The system can answer different types of questions:

1. General conversation questions
2. User-specific questions
3. Date/time questions
4. Topic questions
5. Sentiment questions
6. Statistical questions
7. Conversation summaries

RAG is used whenever conversational evidence is required.

Simple analytical questions are answered directly from the
conversation dataframe when possible.

All generated answers remain grounded in the uploaded chat.
"""

import re
from collections import Counter

import ollama
import pandas as pd

from rag.index import search


MODEL_NAME = "llama3.1:8b"


ANSWER_PROMPT = """
You are the Conversation Intelligence assistant.

You are answering questions about a private conversation.

IMPORTANT RULES:

1. Use ONLY the supplied conversation evidence and analytical
   information.
2. Never invent people, dates, events, statements, or facts.
3. If the evidence is insufficient, clearly say that the
   conversation does not contain enough information.
4. Do not pretend that an inference is a direct statement.
5. When useful, mention the relevant person and date.
6. Keep the answer concise but informative.
7. For summaries, organize the answer into clear bullet points.

Conversation evidence:
{context}

Analytical information:
{analytics}

Question:
{question}

Answer:
"""


def _safe_date(value):
    """Convert a value into pandas Timestamp when possible."""
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def _extract_date_from_question(question: str, df: pd.DataFrame):
    """
    Attempts to identify a date mentioned in the question.

    Supports examples such as:
      15 August
      August 15
      15/08/2025
      15-08-2025
    """

    q = question.lower()

    # Explicit numeric date.
    numeric_match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b",
        q
    )

    if numeric_match:
        day = int(numeric_match.group(1))
        month = int(numeric_match.group(2))
        year = numeric_match.group(3)

        if year:
            year = int(year)

            if year < 100:
                year += 2000

            try:
                return pd.Timestamp(
                    year=year,
                    month=month,
                    day=day
                )
            except Exception:
                pass

    # Month-name date.
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    for month_name, month_number in months.items():
        if month_name not in q:
            continue

        day_match = re.search(
            rf"\b(\d{{1,2}})\s+{month_name}\b",
            q
        )

        if not day_match:
            day_match = re.search(
                rf"\b{month_name}\s+(\d{{1,2}})\b",
                q
            )

        if day_match:
            day = int(day_match.group(1))

            # Use the year represented in the chat.
            dates = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            valid_dates = dates.dropna()

            if not valid_dates.empty:
                year = int(valid_dates.dt.year.mode().iloc[0])

                try:
                    return pd.Timestamp(
                        year=year,
                        month=month_number,
                        day=day
                    )
                except Exception:
                    pass

    return None


def _find_user_in_question(question: str, df: pd.DataFrame):
    """
    Finds a known chat participant mentioned in the question.
    """

    question_lower = question.lower()

    users = [
        str(user)
        for user in df["user"].dropna().unique()
    ]

    # Longest names first to reduce partial matches.
    users.sort(
        key=len,
        reverse=True
    )

    for user in users:
        if user.lower() in question_lower:
            return user

    return None


def _is_summary_question(question: str):
    q = question.lower()

    keywords = [
        "summarize",
        "summary",
        "main things",
        "main topics",
        "what was discussed",
        "what did they discuss",
        "give me an overview",
        "overview of the conversation",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def _is_statistics_question(question: str):
    q = question.lower()

    keywords = [
        "most active",
        "most messages",
        "busiest user",
        "most active user",
        "busiest month",
        "busiest day",
        "total messages",
        "how many messages",
        "how many words",
        "most words",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def _is_topic_question(question: str):
    q = question.lower()

    keywords = [
        "topic",
        "topics",
        "theme",
        "themes",
        "what were they talking about",
        "what did they talk about",
        "discussed about",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def _is_sentiment_question(question: str):
    q = question.lower()

    keywords = [
        "sentiment",
        "positive",
        "negative",
        "neutral",
        "angry",
        "happy",
        "emotion",
        "emotional",
        "tone",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def _direct_statistics_answer(question: str, df: pd.DataFrame):
    """
    Answers simple statistical questions without asking the LLM
    to calculate values from retrieved text.
    """

    q = question.lower()

    if "total messages" in q or "how many messages" in q:
        return (
            f"The conversation contains {len(df):,} messages.",
            []
        )

    if "most active user" in q or "busiest user" in q:
        counts = df["user"].value_counts()

        if counts.empty:
            return "There is not enough data to determine the most active user.", []

        user = counts.index[0]
        count = counts.iloc[0]

        return (
            f"{user} was the most active user with {count:,} messages.",
            []
        )

    if "busiest month" in q:
        counts = df["month"].value_counts()

        if counts.empty:
            return "There is not enough data to determine the busiest month.", []

        month = counts.index[0]

        return (
            f"{month} was the busiest month with {counts.iloc[0]:,} messages.",
            []
        )

    if "busiest day" in q:
        counts = df["day_name"].value_counts()

        if counts.empty:
            return "There is not enough data to determine the busiest day.", []

        day = counts.index[0]

        return (
            f"{day} was the busiest day with {counts.iloc[0]:,} messages.",
            []
        )

    if "how many words" in q:
        word_count = sum(
            len(str(message).split())
            for message in df["message"]
        )

        return (
            f"The conversation contains approximately {word_count:,} words.",
            []
        )

    return None


def _build_date_context(question: str, df: pd.DataFrame):
    """
    Retrieves messages from a specific date when the question
    refers to a particular day.
    """

    target_date = _extract_date_from_question(
        question,
        df
    )

    if target_date is None:
        return None, None

    dates = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    matching = df[
        dates.dt.date == target_date.date()
    ].copy()

    if matching.empty:
        return (
            f"No messages were found for {target_date.strftime('%d %B %Y')}.",
            []
        )

    context_rows = matching.head(100)

    chunks = [{
        "text": "\n".join(
            f"{row['user']}: {row['message']}"
            for _, row in context_rows.iterrows()
        ),
        "start_date": context_rows.iloc[0]["date"],
        "end_date": context_rows.iloc[-1]["date"],
        "users": sorted(
            context_rows["user"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "raw_messages": context_rows.to_dict("records"),
    }]

    analytics = (
        f"{len(matching):,} messages occurred on "
        f"{target_date.strftime('%d %B %Y')}."
    )

    return analytics, chunks


def _build_user_context(
    question: str,
    df: pd.DataFrame
):
    """
    Finds a participant explicitly mentioned in the question.
    """

    user = _find_user_in_question(
        question,
        df
    )

    if user is None:
        return None, None

    user_df = df[
        df["user"].astype(str).str.lower()
        == user.lower()
    ].copy()

    if user_df.empty:
        return None, None

    analytics = (
        f"The question refers specifically to user '{user}'. "
        f"This user has {len(user_df):,} messages in the conversation."
    )

    return analytics, user_df


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No conversation evidence was retrieved."

    return "\n\n---\n\n".join(
        chunk["text"]
        for chunk in chunks
    )


def _generate_answer(
    question: str,
    chunks: list[dict],
    analytics: str = ""
):
    """
    Sends grounded context to the local Ollama model.
    """

    context = _format_context(chunks)

    prompt = ANSWER_PROMPT.format(
        context=context,
        analytics=analytics or "None available.",
        question=question,
    )

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt
    )

    return response["response"].strip()


def answer_question(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 5,
    df: pd.DataFrame | None = None,
    sentiment_data: pd.DataFrame | None = None,
    topic_summary: pd.DataFrame | None = None,
    topic_df: pd.DataFrame | None = None,
) -> dict:
    """
    Main Conversation AI entry point.

    Returns:
        {
            "answer": str,
            "evidence": list[dict],
            "question_type": str
        }
    """

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question about the conversation.",
            "evidence": [],
            "question_type": "empty",
        }

    if df is None:
        relevant_chunks = search(
            question,
            index,
            chunks,
            top_k=top_k
        )

        answer = _generate_answer(
            question,
            relevant_chunks
        )

        return {
            "answer": answer,
            "evidence": relevant_chunks,
            "question_type": "general",
        }

    # ---------------------------------------------------------
    # 1. DIRECT STATISTICS
    # ---------------------------------------------------------

    if _is_statistics_question(question):
        direct_result = _direct_statistics_answer(
            question,
            df
        )

        if direct_result is not None:
            answer, evidence = direct_result

            return {
                "answer": answer,
                "evidence": evidence,
                "question_type": "statistics",
            }

    # ---------------------------------------------------------
    # 2. DATE-SPECIFIC QUESTIONS
    # ---------------------------------------------------------

    date_analytics, date_chunks = _build_date_context(
        question,
        df
    )

    if date_chunks is not None:
        answer = _generate_answer(
            question,
            date_chunks,
            analytics=date_analytics
        )

        return {
            "answer": answer,
            "evidence": date_chunks,
            "question_type": "date",
        }

    # ---------------------------------------------------------
    # 3. USER-SPECIFIC QUESTIONS
    # ---------------------------------------------------------

    user_analytics, user_df = _build_user_context(
        question,
        df
    )

    if user_df is not None:
        # Search using the complete question so the semantic model
        # understands both the person and subject.
        relevant_chunks = search(
            question,
            index,
            chunks,
            top_k=top_k * 2
        )

        # Prefer chunks that actually contain the requested user.
        user_lower = _find_user_in_question(
            question,
            df
        ).lower()

        user_chunks = [
            chunk
            for chunk in relevant_chunks
            if any(
                user_lower == str(u).lower()
                for u in chunk.get("users", [])
            )
        ]

        if user_chunks:
            relevant_chunks = user_chunks[:top_k]
        else:
            relevant_chunks = relevant_chunks[:top_k]

        answer = _generate_answer(
            question,
            relevant_chunks,
            analytics=user_analytics
        )

        return {
            "answer": answer,
            "evidence": relevant_chunks,
            "question_type": "user",
        }

    # ---------------------------------------------------------
    # 4. TOPIC QUESTIONS
    # ---------------------------------------------------------

    if _is_topic_question(question):

        analytics_parts = []

        if (
            topic_summary is not None
            and not topic_summary.empty
        ):
            topic_lines = []

            for _, row in topic_summary.iterrows():
                topic_lines.append(
                    f"Topic {row['topic_id']}: "
                    f"{row['label']} "
                    f"({row['message_count']} messages)"
                )

            analytics_parts.append(
                "Discovered topics:\n"
                + "\n".join(topic_lines)
            )

        relevant_chunks = search(
            question,
            index,
            chunks,
            top_k=top_k
        )

        answer = _generate_answer(
            question,
            relevant_chunks,
            analytics="\n\n".join(analytics_parts)
        )

        return {
            "answer": answer,
            "evidence": relevant_chunks,
            "question_type": "topic",
        }

    # ---------------------------------------------------------
    # 5. SENTIMENT QUESTIONS
    # ---------------------------------------------------------

    if _is_sentiment_question(question):

        analytics_parts = []

        if (
            sentiment_data is not None
            and not sentiment_data.empty
        ):
            if "transformer_value" in sentiment_data.columns:
                values = pd.to_numeric(
                    sentiment_data["transformer_value"],
                    errors="coerce"
                ).dropna()

                if not values.empty:
                    positive = int((values == 1).sum())
                    neutral = int((values == 0).sum())
                    negative = int((values == -1).sum())

                    analytics_parts.append(
                        "Transformer sentiment distribution: "
                        f"{positive} positive, "
                        f"{neutral} neutral, "
                        f"{negative} negative."
                    )

            if "vader_value" in sentiment_data.columns:
                values = pd.to_numeric(
                    sentiment_data["vader_value"],
                    errors="coerce"
                ).dropna()

                if not values.empty:
                    positive = int((values == 1).sum())
                    neutral = int((values == 0).sum())
                    negative = int((values == -1).sum())

                    analytics_parts.append(
                        "VADER sentiment distribution: "
                        f"{positive} positive, "
                        f"{neutral} neutral, "
                        f"{negative} negative."
                    )

        relevant_chunks = search(
            question,
            index,
            chunks,
            top_k=top_k
        )

        answer = _generate_answer(
            question,
            relevant_chunks,
            analytics="\n".join(analytics_parts)
        )

        return {
            "answer": answer,
            "evidence": relevant_chunks,
            "question_type": "sentiment",
        }

    # ---------------------------------------------------------
    # 6. GENERAL / SUMMARY QUESTIONS
    # ---------------------------------------------------------

    relevant_chunks = search(
        question,
        index,
        chunks,
        top_k=top_k
    )

    analytics = ""

    if _is_summary_question(question):
        analytics = (
            f"The conversation contains {len(df):,} messages "
            f"from {df['user'].nunique():,} users."
        )

    answer = _generate_answer(
        question,
        relevant_chunks,
        analytics=analytics
    )

    return {
        "answer": answer,
        "evidence": relevant_chunks,
        "question_type": "general",
    }


def format_evidence_for_display(
    evidence: list[dict]
) -> str:
    """
    Formats evidence as readable source messages.

    Duplicate messages are removed because overlapping chunks
    intentionally contain some of the same messages.
    """

    lines = []
    seen = set()

    for chunk in evidence:

        for msg in chunk.get("raw_messages", []):

            date_value = msg.get("date")
            date = _safe_date(date_value)

            if date is not None:
                date_str = date.strftime(
                    "%d %b %Y"
                )
            else:
                date_str = str(date_value)

            user = str(
                msg.get("user", "Unknown")
            )

            message = str(
                msg.get("message", "")
            ).strip()

            key = (
                date_str,
                user,
                message
            )

            if key in seen:
                continue

            seen.add(key)

            snippet = message[:180]

            lines.append(
                f"{date_str} · {user}\n"
                f"{snippet}"
            )

    return "\n\n".join(lines)