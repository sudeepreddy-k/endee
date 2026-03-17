# AI Legal Assistant using RAG + Endee Vector Database

This project demonstrates a full **Retrieval-Augmented Generation (RAG)** system built for the legal domain. It uses the **Endee Vector Database** workflow to retrieve relevant Indian Penal Code (IPC) sections and provides a synthesized AI analysis for user scenarios.

## Project Overview

The project demonstrates a practical use case of **Semantic Search**. It allows users to insert text snippets into the Endee database and query them using semantic similarity, rather than exact keyword matching.

The application uses:
1. **Endee Vector Database** as the high-performance retrieval layer.
2. **all-MiniLM-L6-v2** from HuggingFace (`sentence-transformers`) to generate 384-dimensional dense semantic vectors.
3. **Streamlit** to provide an interactive Web UI for inserting documents and searching linearly.
4. **Endee Python SDK** to connect to the database, index data, and query via cosine similarity.

## System Design

1. **Embedding Stage**: When text is supplied via the UI, `sentence-transformers` vectorizes it into a list of 384 floats.
2. **Indexing Stage**: The vectors, along with the original text as payload metadata, are upserted into an Endee index named `semantic_search`. The space type is configured for cosine distance calculations using `INT8` precision.
3. **Retrieval Stage**: A semantic search query is also embedded. The Endee database mathematically returns the closest Top-K documents based on similarity scores.

## Setup Instructions

### Prerequisites
- Docker & Docker Compose
- Python 3.8+

### 1. Requirements
Ensure you have Python 3.8+ installed.

### 2. Install Python Dependencies
It is recommended to use a virtual environment:

```bash
cd ai_project
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application
Launch the semantic search app:

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## IPC Semantic Search Features
- **Zero-Docker Setup**: Uses FAISS for local vector storage, ensuring the app runs without external dependencies.
- **Premium UI**: Custom dark theme with similarity score visualization.
- **Pre-indexed Data**: Automatically indices 63 IPC sections on startup.
- **Natural Language Querying**: Allows matching legal sections based on descriptions of crimes.

---
*(Note: The original Endee documentation `README_ENDEE.md` is preserved in the root directory for reference).*
