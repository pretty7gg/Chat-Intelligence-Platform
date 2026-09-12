# """
# app.py

# Main Streamlit app. Structure, top to bottom:

#   1. Upload + parse (using the WhatsAppParser)
#   2. Top statistics (unchanged from original)
#   3. Timeline / activity map (unchanged from original)
#   4. Sentiment: VADER vs transformer, side by side
#   5. Topics: embeddings + clustering (replaces "most common words")
#   6. Ask Your Chat: RAG question-answering with citations (the centerpiece)

# Everything runs locally - no paid API calls anywhere in this file.
# """

# import os
# # Fix for a common Mac + Anaconda crash: PyTorch and NumPy/scikit-learn
# # (via Anaconda's MKL) each bundle their own OpenMP threading library.
# # When both load in the same process it segfaults. This must be set
# # BEFORE torch/transformers are imported anywhere.
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# import helper
# from parsers.whatsapp_parser import WhatsAppParser
# from nlp.sentiment import add_sentiment_columns
# from nlp.topics import discover_topics
# from rag.index import chunk_messages, build_index
# from rag.qa import answer_question, format_evidence_for_display

# st.set_page_config(page_title="Conversation Intelligence", layout="wide")

# # Hides Streamlit's default menu (Deploy, Rerun, Record screen, etc.) and
# # the "Made with Streamlit" footer, so the UI reads as a real product
# # screen rather than an obvious framework default.
# st.markdown(
#     """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.sidebar.title("Conversation Intelligence Platform")
# uploaded_file = st.sidebar.file_uploader("Choose a chat export file (.txt)")

# st.markdown(
#     "<h1 style='text-align: center; color: grey;'>Conversation Intelligence Platform</h1>",
#     unsafe_allow_html=True,
# )

# # --- Session state so we don't recompute everything on every interaction ---
# if "data" not in st.session_state:
#     st.session_state.data = None
# if "rag_index" not in st.session_state:
#     st.session_state.rag_index = None
# if "rag_chunks" not in st.session_state:
#     st.session_state.rag_chunks = None


# if uploaded_file is not None:
#     bytes_data = uploaded_file.getvalue()
#     raw_text = bytes_data.decode("utf-8")

#     # Detect a NEW file upload (different name or content) and reset all
#     # cached state. Without this, uploading a second chat silently keeps
#     # using the first chat's parsed data, sentiment results, topics, and
#     # RAG index - the app looks like it's "ignoring" the new upload.
#     file_fingerprint = (uploaded_file.name, len(bytes_data))
#     if st.session_state.get("file_fingerprint") != file_fingerprint:
#         st.session_state.file_fingerprint = file_fingerprint
#         st.session_state.data = None
#         st.session_state.rag_index = None
#         st.session_state.rag_chunks = None
#         st.session_state.pop("sentiment_data", None)
#         st.session_state.pop("topic_summary", None)
#         st.session_state.pop("topic_df", None)

#     if st.session_state.data is None:
#         with st.spinner("Parsing chat..."):
#             st.session_state.data = WhatsAppParser().parse(raw_text)

#     data = st.session_state.data

#     if data.empty:
#         st.error(
#             "No messages could be parsed from this file. This usually means "
#             "the date/time format in your export doesn't match what the "
#             "parser expects (e.g. iPhone exports often use a different "
#             "format than Android). Paste the first 2-3 lines of your raw "
#             ".txt file to Claude to get the parser regex adjusted."
#         )
#         st.stop()

#     user_list = sorted(data["user"].unique().tolist())
#     user_list.insert(0, "Overall")
#     selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

#     tab_stats, tab_sentiment, tab_topics, tab_ask = st.tabs(
#         ["Overview", "Sentiment", "Topics", "Ask Your Chat"]
#     )

#     # ---------------- TAB 1: Overview (stats/timeline/activity) ----------------
#     with tab_stats:
#         num_messages, words, num_media, num_links, num_deleted = helper.fetch_stats(
#             selected_user, data
#         )
#         col1, col2, col3, col4, col5 = st.columns(5)
#         col1.metric("Total Messages", num_messages)
#         col2.metric("Total Words", words)
#         col3.metric("Media Shared", num_media)
#         col4.metric("Links Shared", num_links)
#         col5.metric("Messages Deleted", num_deleted)

#         st.subheader("Monthly Timeline")
#         timeline = helper.monthly_timeline(selected_user, data)
#         fig, ax = plt.subplots()
#         ax.plot(timeline["time"], timeline["message"], color="green")
#         plt.xticks(rotation="vertical")
#         st.pyplot(fig)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("Most Busy Day")
#             busy_day = helper.week_activity_map(selected_user, data)
#             fig, ax = plt.subplots()
#             ax.bar(busy_day.index, busy_day.values, color="purple")
#             plt.xticks(rotation="vertical")
#             st.pyplot(fig)
#         with col2:
#             st.subheader("Most Busy Month")
#             busy_month = helper.month_activity_map(selected_user, data)
#             fig, ax = plt.subplots()
#             ax.bar(busy_month.index, busy_month.values, color="orange")
#             plt.xticks(rotation="vertical")
#             st.pyplot(fig)

#         st.subheader("Weekly Activity Heatmap")
#         heatmap = helper.activity_heatmap(selected_user, data)
#         fig, ax = plt.subplots()
#         sns.heatmap(heatmap, ax=ax)
#         st.pyplot(fig)

#         if selected_user == "Overall":
#             st.subheader("Most Busy Users")
#             top_counts, percent_df = helper.most_busy_users(data)
#             col1, col2 = st.columns(2)
#             with col1:
#                 fig, ax = plt.subplots()
#                 ax.bar(top_counts.index, top_counts.values, color="red")
#                 plt.xticks(rotation="vertical")
#                 st.pyplot(fig)
#             with col2:
#                 st.dataframe(percent_df)

#         st.subheader("Wordcloud")
#         with open("stop_hinglish.txt", "r") as f:
#             stop_words = f.read()
#         wc = helper.create_wordcloud(selected_user, data, stop_words)
#         fig, ax = plt.subplots()
#         ax.imshow(wc)
#         ax.axis("off")
#         st.pyplot(fig)

#     # ---------------- TAB 2: Sentiment ----------------
#     with tab_sentiment:
#         st.write(
#             "VADER is a fast lexicon-based baseline. The transformer model "
#             "understands context/negation better but is slower. Both are "
#             "shown so you can see where they agree or disagree."
#         )
#         run_transformer = st.checkbox(
#             "Also run transformer model (slower, more accurate on context)",
#             value=False,
#         )

#         if st.button("Run Sentiment Analysis"):
#             with st.spinner("Analyzing sentiment..."):
#                 sentiment_data = add_sentiment_columns(
#                     data, use_transformer=run_transformer
#                 )
#             st.session_state.sentiment_data = sentiment_data

#         if "sentiment_data" in st.session_state:
#             sdata = st.session_state.sentiment_data
#             column_to_use = (
#                 "transformer_value" if run_transformer else "vader_value"
#             )

#             if selected_user == "Overall":
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.markdown("**Most Positive Contribution**")
#                     st.dataframe(helper.sentiment_percentage(sdata, column_to_use, 1))
#                 with col2:
#                     st.markdown("**Most Neutral Contribution**")
#                     st.dataframe(helper.sentiment_percentage(sdata, column_to_use, 0))
#                 with col3:
#                     st.markdown("**Most Negative Contribution**")
#                     st.dataframe(helper.sentiment_percentage(sdata, column_to_use, -1))

#             if run_transformer:
#                 agreement = (
#                     sdata["vader_value"] == sdata["transformer_value"]
#                 ).mean()
#                 st.info(
#                     f"VADER and the transformer model agree on "
#                     f"{agreement:.1%} of messages in this chat."
#                 )

#                 # Show WHERE they disagree, not just the summary number.
#                 # This is what makes the agreement stat defensible rather
#                 # than a bare claim - see the earlier discussion on
#                 # citations/evidence applied to sentiment instead of RAG.
#                 label_names = {1: "positive", 0: "neutral", -1: "negative"}
#                 disagreements = sdata[
#                     sdata["vader_value"] != sdata["transformer_value"]
#                 ].copy()

#                 with st.expander(
#                     f"See {len(disagreements)} messages where VADER and "
#                     f"the transformer disagreed"
#                 ):
#                     if disagreements.empty:
#                         st.write("No disagreements found.")
#                     else:
#                         display_df = disagreements[
#                             ["date", "user", "message", "vader_value", "transformer_value"]
#                         ].head(20).copy()
#                         display_df["vader_value"] = display_df["vader_value"].map(label_names)
#                         display_df["transformer_value"] = display_df["transformer_value"].map(label_names)
#                         display_df.columns = [
#                             "date", "user", "message", "VADER said", "Transformer said"
#                         ]
#                         st.dataframe(display_df, use_container_width=True)

#     # ---------------- TAB 3: Topics ----------------
#     with tab_topics:
#         st.write(
#             "Instead of just showing common words, this groups messages by "
#             "MEANING using embeddings, then labels each group by its "
#             "distinctive terms."
#         )
#         n_topics = st.slider("Number of topics to look for", 3, 12, 6)

#         if st.button("Discover Topics"):
#             with st.spinner("Embedding messages and clustering..."):
#                 topic_df, topic_summary = discover_topics(data, n_topics=n_topics)
#             st.session_state.topic_summary = topic_summary
#             st.session_state.topic_df = topic_df

#         if "topic_summary" in st.session_state:
#             st.subheader("Discovered Topics")
#             st.dataframe(st.session_state.topic_summary)

#             selected_topic = st.selectbox(
#                 "View messages from a topic",
#                 st.session_state.topic_summary["topic_id"].tolist(),
#             )
#             topic_messages = st.session_state.topic_df[
#                 st.session_state.topic_df["topic_id"] == selected_topic
#             ][["date", "user", "message"]]
#             st.dataframe(topic_messages)

#     # ---------------- TAB 4: Ask Your Chat (RAG) ----------------
#     with tab_ask:
#         st.write(
#             "Ask a question about this conversation. The answer is grounded "
#             "in real messages, which are shown below as evidence."
#         )

#         if st.session_state.rag_index is None:
#             if st.button("Build Index (do this once per chat)"):
#                 with st.spinner("Chunking and embedding messages..."):
#                     chunks = chunk_messages(data)
#                     index, _ = build_index(chunks)
#                     st.session_state.rag_index = index
#                     st.session_state.rag_chunks = chunks
#                 st.success("Index built. You can now ask questions below.")

#         if st.session_state.rag_index is not None:
#             question = st.text_input("Ask something about this chat...")
#             if question and st.button("Get Answer"):
#                 with st.spinner("Thinking..."):
#                     result = answer_question(
#                         question,
#                         st.session_state.rag_index,
#                         st.session_state.rag_chunks,
#                     )
#                 st.markdown(f"**Answer:** {result['answer']}")
#                 st.markdown("**Evidence**")
#                 st.text(format_evidence_for_display(result["evidence"]))
# else:
#     st.info("Upload a chat export file to get started.")


"""
app.py

Main Streamlit application.

Features:
1. Upload + parse WhatsApp conversation
2. Conversation statistics
3. Timeline / activity analysis
4. Sentiment analysis
5. Topic discovery
6. Conversation AI / RAG

The RAG index is built automatically when a new conversation
is uploaded. The user never needs to manually create an index.
"""

import os

# Mac + Anaconda / PyTorch OpenMP stability.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import helper

from parsers.whatsapp_parser import WhatsAppParser

from nlp.sentiment import add_sentiment_columns
from nlp.topics import discover_topics

from rag.index import chunk_messages, build_index
from rag.qa import (
    answer_question,
    format_evidence_for_display,
)


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Conversation Intelligence",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title(
    "Conversation Intelligence Platform"
)

uploaded_file = st.sidebar.file_uploader(
    "Choose a chat export file (.txt)",
    type=["txt"]
)


# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align: center; color: grey;'>
        Conversation Intelligence Platform
    </h1>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = None

if "file_fingerprint" not in st.session_state:
    st.session_state.file_fingerprint = None

if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = None

if "topic_summary" not in st.session_state:
    st.session_state.topic_summary = None

if "topic_df" not in st.session_state:
    st.session_state.topic_df = None


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------

if uploaded_file is not None:

    bytes_data = uploaded_file.getvalue()

    try:
        raw_text = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = bytes_data.decode(
            "utf-8",
            errors="replace"
        )

    # --------------------------------------------------------
    # DETECT NEW FILE
    # --------------------------------------------------------

    file_fingerprint = (
        uploaded_file.name,
        len(bytes_data)
    )

    if (
        st.session_state.file_fingerprint
        != file_fingerprint
    ):

        st.session_state.file_fingerprint = (
            file_fingerprint
        )

        # Reset all conversation-specific state.
        st.session_state.data = None
        st.session_state.rag_index = None
        st.session_state.rag_chunks = None

        st.session_state.sentiment_data = None
        st.session_state.topic_summary = None
        st.session_state.topic_df = None

    # --------------------------------------------------------
    # PARSE CHAT
    # --------------------------------------------------------

    if st.session_state.data is None:

        with st.spinner("Parsing conversation..."):

            st.session_state.data = (
                WhatsAppParser().parse(raw_text)
            )

    data = st.session_state.data

    # --------------------------------------------------------
    # EMPTY DATA CHECK
    # --------------------------------------------------------

    if data.empty:

        st.error(
            "No messages could be parsed from this file. "
            "This usually means the date/time format in your "
            "export doesn't match what the parser expects."
        )

        st.stop()

    # --------------------------------------------------------
    # AUTOMATIC RAG INDEX CREATION
    # --------------------------------------------------------

    if st.session_state.rag_index is None:

        with st.spinner(
            "Preparing Conversation AI..."
        ):

            rag_chunks = chunk_messages(
                data,
                chunk_size=5,
                overlap=2
            )

            rag_index, _ = build_index(
                rag_chunks
            )

            st.session_state.rag_chunks = (
                rag_chunks
            )

            st.session_state.rag_index = (
                rag_index
            )

        st.success(
            "✓ Conversation AI is ready."
        )

    # --------------------------------------------------------
    # USER FILTER
    # --------------------------------------------------------

    user_list = sorted(
        data["user"].astype(str).unique().tolist()
    )

    user_list.insert(
        0,
        "Overall"
    )

    selected_user = st.sidebar.selectbox(
        "Show analysis wrt",
        user_list
    )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    (
        tab_stats,
        tab_sentiment,
        tab_topics,
        tab_ask
    ) = st.tabs(
        [
            "Overview",
            "Sentiment",
            "Topics",
            "Ask Your Chat",
        ]
    )

    # ========================================================
    # TAB 1 — OVERVIEW
    # ========================================================

    with tab_stats:

        (
            num_messages,
            words,
            num_media,
            num_links,
            num_deleted,
        ) = helper.fetch_stats(
            selected_user,
            data
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Total Messages",
            num_messages
        )

        col2.metric(
            "Total Words",
            words
        )

        col3.metric(
            "Media Shared",
            num_media
        )

        col4.metric(
            "Links Shared",
            num_links
        )

        col5.metric(
            "Messages Deleted",
            num_deleted
        )

        # ----------------------------------------------------
        # MONTHLY TIMELINE
        # ----------------------------------------------------

        st.subheader(
            "Monthly Timeline"
        )

        timeline = helper.monthly_timeline(
            selected_user,
            data
        )

        fig, ax = plt.subplots()

        ax.plot(
            timeline["time"],
            timeline["message"],
            color="green"
        )

        plt.xticks(
            rotation="vertical"
        )

        st.pyplot(
            fig
        )

        # ----------------------------------------------------
        # BUSY DAY / MONTH
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Most Busy Day"
            )

            busy_day = helper.week_activity_map(
                selected_user,
                data
            )

            fig, ax = plt.subplots()

            ax.bar(
                busy_day.index,
                busy_day.values,
                color="purple"
            )

            plt.xticks(
                rotation="vertical"
            )

            st.pyplot(fig)

        with col2:

            st.subheader(
                "Most Busy Month"
            )

            busy_month = helper.month_activity_map(
                selected_user,
                data
            )

            fig, ax = plt.subplots()

            ax.bar(
                busy_month.index,
                busy_month.values,
                color="orange"
            )

            plt.xticks(
                rotation="vertical"
            )

            st.pyplot(fig)

        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        st.subheader(
            "Weekly Activity Heatmap"
        )

        heatmap = helper.activity_heatmap(
            selected_user,
            data
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            heatmap,
            ax=ax
        )

        st.pyplot(fig)

        # ----------------------------------------------------
        # MOST BUSY USERS
        # ----------------------------------------------------

        if selected_user == "Overall":

            st.subheader(
                "Most Busy Users"
            )

            top_counts, percent_df = (
                helper.most_busy_users(data)
            )

            col1, col2 = st.columns(2)

            with col1:

                fig, ax = plt.subplots()

                ax.bar(
                    top_counts.index,
                    top_counts.values,
                    color="red"
                )

                plt.xticks(
                    rotation="vertical"
                )

                st.pyplot(fig)

            with col2:

                st.dataframe(
                    percent_df,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # WORD CLOUD
        # ----------------------------------------------------

        st.subheader(
            "Wordcloud"
        )

        with open(
            "stop_hinglish.txt",
            "r",
            encoding="utf-8"
        ) as f:
            stop_words = f.read()

        wc = helper.create_wordcloud(
            selected_user,
            data,
            stop_words
        )

        fig, ax = plt.subplots()

        ax.imshow(wc)
        ax.axis("off")

        st.pyplot(fig)

    # ========================================================
    # TAB 2 — SENTIMENT
    # ========================================================

    with tab_sentiment:

        st.write(
            "VADER is a fast lexicon-based baseline. "
            "The transformer model understands context "
            "and negation better but is slower."
        )

        run_transformer = st.checkbox(
            "Also run transformer model "
            "(slower, more context-aware)",
            value=False
        )

        if st.button(
            "Run Sentiment Analysis"
        ):

            with st.spinner(
                "Analyzing sentiment..."
            ):

                sentiment_data = (
                    add_sentiment_columns(
                        data,
                        use_transformer=run_transformer
                    )
                )

            st.session_state.sentiment_data = (
                sentiment_data
            )

        if st.session_state.sentiment_data is not None:

            sdata = (
                st.session_state.sentiment_data
            )

            column_to_use = (
                "transformer_value"
                if run_transformer
                else "vader_value"
            )

            if selected_user == "Overall":

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.markdown(
                        "**Most Positive Contribution**"
                    )

                    st.dataframe(
                        helper.sentiment_percentage(
                            sdata,
                            column_to_use,
                            1
                        )
                    )

                with col2:

                    st.markdown(
                        "**Most Neutral Contribution**"
                    )

                    st.dataframe(
                        helper.sentiment_percentage(
                            sdata,
                            column_to_use,
                            0
                        )
                    )

                with col3:

                    st.markdown(
                        "**Most Negative Contribution**"
                    )

                    st.dataframe(
                        helper.sentiment_percentage(
                            sdata,
                            column_to_use,
                            -1
                        )
                    )

            if run_transformer:

                agreement = (
                    sdata["vader_value"]
                    == sdata["transformer_value"]
                ).mean()

                st.info(
                    f"VADER and the transformer model "
                    f"agree on {agreement:.1%} of messages."
                )

                label_names = {
                    1: "positive",
                    0: "neutral",
                    -1: "negative"
                }

                disagreements = sdata[
                    sdata["vader_value"]
                    != sdata["transformer_value"]
                ].copy()

                with st.expander(
                    f"See {len(disagreements)} messages "
                    "where the models disagreed"
                ):

                    if disagreements.empty:

                        st.write(
                            "No disagreements found."
                        )

                    else:

                        display_df = disagreements[
                            [
                                "date",
                                "user",
                                "message",
                                "vader_value",
                                "transformer_value",
                            ]
                        ].head(20).copy()

                        display_df[
                            "vader_value"
                        ] = display_df[
                            "vader_value"
                        ].map(label_names)

                        display_df[
                            "transformer_value"
                        ] = display_df[
                            "transformer_value"
                        ].map(label_names)

                        display_df.columns = [
                            "date",
                            "user",
                            "message",
                            "VADER said",
                            "Transformer said",
                        ]

                        st.dataframe(
                            display_df,
                            use_container_width=True
                        )

    # ========================================================
    # TAB 3 — TOPICS
    # ========================================================

    with tab_topics:

        st.write(
            "Messages are grouped by semantic meaning "
            "using sentence embeddings and K-Means. "
            "TF-IDF terms are then used to describe each cluster."
        )

        n_topics = st.slider(
            "Number of topics to look for",
            3,
            12,
            6
        )

        if st.button(
            "Discover Topics"
        ):

            with st.spinner(
                "Discovering conversation topics..."
            ):

                topic_df, topic_summary = (
                    discover_topics(
                        data,
                        n_topics=n_topics
                    )
                )

            st.session_state.topic_summary = (
                topic_summary
            )

            st.session_state.topic_df = (
                topic_df
            )

        if (
            st.session_state.topic_summary
            is not None
        ):

            st.subheader(
                "Discovered Topics"
            )

            st.dataframe(
                st.session_state.topic_summary,
                use_container_width=True
            )

            selected_topic = st.selectbox(
                "View messages from a topic",
                st.session_state.topic_summary[
                    "topic_id"
                ].tolist()
            )

            topic_messages = (
                st.session_state.topic_df[
                    st.session_state.topic_df[
                        "topic_id"
                    ] == selected_topic
                ][
                    [
                        "date",
                        "user",
                        "message"
                    ]
                ]
            )

            st.dataframe(
                topic_messages,
                use_container_width=True
            )

    # ========================================================
    # TAB 4 — CONVERSATION AI
    # ========================================================

    with tab_ask:

        st.markdown(
            """
            ### 🤖 Conversation AI

            Ask questions about your conversation and get
            evidence-backed answers from the actual messages.
            """
        )

        # ----------------------------------------------------
        # CONVERSATION SUMMARY CARD
        # ----------------------------------------------------

        summary_col1, summary_col2, summary_col3 = (
            st.columns(3)
        )

        summary_col1.metric(
            "Messages",
            f"{len(data):,}"
        )

        summary_col2.metric(
            "Participants",
            f"{data['user'].nunique():,}"
        )

        summary_col3.metric(
            "Words",
            f"{sum(len(str(x).split()) for x in data['message']):,}"
        )

        st.divider()

        # ----------------------------------------------------
        # SUGGESTED QUESTIONS
        # ----------------------------------------------------

        st.markdown(
            "### 💡 Try asking"
        )

        suggested_questions = [
            "What were the main topics discussed?",
            "Who was the most active user?",
            "What happened on 15 August?",
            "Summarize the conversation.",
            "What did the most active user mainly talk about?",
            "What was the overall sentiment?",
        ]

        selected_suggestion = st.selectbox(
            "Suggested questions",
            [
                "Choose a question..."
            ] + suggested_questions
        )

        # ----------------------------------------------------
        # QUESTION INPUT
        # ----------------------------------------------------

        question = st.text_area(
            "Ask your conversation",
            value=(
                ""
                if selected_suggestion
                == "Choose a question..."
                else selected_suggestion
            ),
            placeholder=(
                "e.g. What did Rahul say about the project?"
            ),
            height=100
        )

        ask_button = st.button(
            "Ask",
            type="primary"
        )

        if ask_button:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Analyzing the conversation..."
                ):

                    result = answer_question(
                        question=question,
                        index=st.session_state.rag_index,
                        chunks=st.session_state.rag_chunks,
                        top_k=5,
                        df=data,
                        sentiment_data=(
                            st.session_state.sentiment_data
                        ),
                        topic_summary=(
                            st.session_state.topic_summary
                        ),
                        topic_df=(
                            st.session_state.topic_df
                        ),
                    )

                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.markdown(
                    "### 🧠 Answer"
                )

                st.write(
                    result["answer"]
                )

                # ------------------------------------------------
                # QUESTION TYPE
                # ------------------------------------------------

                question_type = result.get(
                    "question_type",
                    "general"
                )

                st.caption(
                    f"Analysis type: {question_type}"
                )

                # ------------------------------------------------
                # EVIDENCE
                # ------------------------------------------------

                evidence = result.get(
                    "evidence",
                    []
                )

                if evidence:

                    st.markdown(
                        "### 📚 Evidence"
                    )

                    st.caption(
                        "These are the conversation messages "
                        "used to support the answer."
                    )

                    evidence_text = (
                        format_evidence_for_display(
                            evidence
                        )
                    )

                    st.text(
                        evidence_text
                    )

                else:

                    st.info(
                        "This answer was calculated directly "
                        "from the conversation analytics."
                    )

else:

    st.info(
        "Upload a chat export file to get started."
    )