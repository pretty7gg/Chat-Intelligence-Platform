# 💬 Converso AI — Conversation Intelligence Platform

A local-first Python/Streamlit platform that turns raw WhatsApp chat exports into statistical, sentiment, topic, and conversational-AI insights . Everything, including the LLM used for Q&A, runs on your own machine.

---

## 🚀 Key Features

### 📊 Conversation Analytics
- Message, word, media, and link counts — overall or per user
- Monthly timeline, weekly activity heatmap, busiest day/month
- Most active users with contribution percentages


### 🙂 Sentiment Analysis
- Sentiment scoring using both **VADER** (lexicon-based) and a local **transformer model**, with an agreement rate shown between the two
- Includes a small evaluation script (`eval/`) to check model accuracy against human-verified labels

### 🧠 Topic Discovery
- Groups messages by meaning (via embeddings + clustering) instead of just common words
- Each topic auto-labeled using TF-IDF top terms

### 🤖 Ask Your Chat — Retrieval-Augmented Q&A
- Chat history is chunked (overlapping windows), embedded, and indexed in **FAISS**
- A local LLM (**Llama 3.2 / 3.1 via Ollama**) answers questions grounded *only* in retrieved evidence — no hallucinated facts
- **6-way intelligent query routing** decides how to answer each question:

| Question Type | Example | How it's Answered |
|---|---|---|
| Statistics | "Who is the most active user?" | Direct pandas computation |
| Date-specific | "What happened on 15 August?" | Filtered message retrieval |
| User-specific | "What did Rahul talk about?" | User-filtered semantic search |
| Topic | "What were the main topics?" | Topic-summary + retrieval |
| Sentiment | "What was the overall mood?" | Sentiment-distribution + retrieval |
| General / Summary | "Summarize the conversation." | Full RAG pipeline |

- Every answer comes with **citation-backed evidence**: the exact messages, dates, and users the answer was grounded in

---

## 🏗️ Architecture

```mermaid
flowchart TB
    A[WhatsApp .txt Export] --> B[WhatsAppParser]
    B --> C[Standardized DataFrame<br/>date · user · message]

    C --> D[Overview Tab<br/>Stats · Timeline · Heatmap · Wordcloud]

    C --> E[Sentiment Engine]
    E --> E1[VADER<br/>Lexicon-based]
    E --> E2[Transformer<br/>RoBERTa - Twitter]
    E1 --> E3[Agreement Analysis]
    E2 --> E3

    C --> F[Topic Discovery]
    F --> F1[Sentence-Transformer<br/>Embeddings]
    F1 --> F2[K-Means Clustering]
    F2 --> F3[TF-IDF Auto-Labeling]

    C --> G[RAG Pipeline]
    G --> G1[Chunking<br/>overlapping windows]
    G1 --> G2[Embedding<br/>all-MiniLM-L6-v2]
    G2 --> G3[FAISS Vector Index]
    G3 --> G4[Query Router<br/>6-way classification]
    G4 --> G5[Local LLM<br/>Llama 3.2 via Ollama]
    G5 --> G6[Grounded Answer<br/>+ Evidence Citations]

    D --> H[Streamlit UI]
    E3 --> H
    F3 --> H
    G6 --> H
```




---

<!-- ## 📸 Screenshots


| Overview | Sentiment | Topic Discovery | Ask Your Chat |
|---|---|---|---|
| ![Overview](screenshots/overview/overview2.png) | ![Sentiment](screenshots/sentiment.png) | ![Topic Discovery](screenshots/topic.png) | ![Q&A Chatbot](screenshots/ask.png) | -->

## 📸 Screenshots

### Overview
*(scrollable page — top and bottom halves shown separately)*

![Overview - Stats & Timeline](screenshots/overview/overview1.png)
![Overview - Heatmap & Wordcloud](screenshots/overview/overview2.png)

### Sentiment Analysis
![Sentiment](screenshots/sentiment.png)

### Topic Discovery
![Topics](screenshots/topic.png)

### Q&A ChatBot
![Ask Your Chat](screenshots/ask.png)



