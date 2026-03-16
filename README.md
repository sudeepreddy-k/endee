# Semantic Search with Endee Vector Database

This repository is submitted as an assignment to demonstrate building an AI/ML project using the [Endee](https://endee.io/) Vector Database. The original Endee codebase is preserved, and a demonstration application has been added under the `ai_project/` directory.

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

### 1. Start the Endee Vector Database
Navigate to the `ai_project` directory and start the Endee server using Docker Compose:

```bash
cd ai_project
docker-compose up -d
```
> Wait a moment for the server to initialize on `http://localhost:8080`.

### 2. Install Python Dependencies
It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
Launch the semantic search app:

```bash
streamlit run app.py
```

The application will open in your default browser automatically. You can start by adding knowledge chunks on the first section and then searching for concepts in the second section!

---
*(Note: The original Endee documentation `README_ENDEE.md` is preserved in the root directory for reference).*
