# RAG-Powered Multi-Agent Q&A Assistant

## Overview

This project implements a **RAG-based (Retrieval-Augmented Generation)** assistant enhanced with **agentic logic** to intelligently handle user queries. It combines document retrieval with local LLM-based generation and dynamically routes queries based on intent (e.g., calculation, definition, or information retrieval).

You can access the webapp using this link:
https://multi-agent-q-a-ksharma.streamlit.app/

If the LLM is offline please Email me at: kshitijsharma1106@gmail.com, the model is hosted locally and may not be up 24x7.

---

## Architecture Summary

### Components

- **Document Ingestion & Chunking**: Loads and processes `.txt` files into embeddings.
- **Vector Store**: Built using FAISS with SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
- **LLM**: A quantized local model (`mistral-7b-instruct-v0.1.Q2_K.gguf`) loaded via `llama-cpp`.
- **Agentic Workflow**: Uses keyword detection to dispatch queries:
  - `calculate` → Calculator Tool
  - `define` / `meaning of` → Dictionary Tool (WordNet)
  - Otherwise → RAG → LLM
- **Frontend Interface**: A Streamlit app for interaction, with context and routing transparency.

---

## Deliverables (as per assignment)

### 1. Data Ingestion

- Loads all `.txt` files from `Documents/`.
- Uses `TextLoader` from LangChain and `RecursiveCharacterTextSplitter` for chunking (300 chars, 50 overlap).

### 2. Vector Store & Retrieval

- Uses **FAISS** for vector indexing and retrieval.
- Embeddings via `SentenceTransformer("all-MiniLM-L6-v2")`.
- Top 3 chunks are returned for a query using `similarity_search()`.

### 3. LLM Integration

- Local model: **Mistral-7B-Instruct GGUF** via `llama-cpp`.
- Automatically downloaded on first run via `download_model.py`.
- Prompt templating is used to generate final answers based on retrieved context.

### 4. Agentic Workflow

- Implemented in `agent_rag_pipeline.py` and `app.py`:
  - Queries with:
    - **"calculate" / "compute"** → routed to `mock_calculator` (uses `eval`)
    - **"define" / "meaning of"** → routed to `mock_dictionary` (WordNet via NLTK)
    - Others → Routed through RAG + LLM
- Every decision is logged and shown in the UI for traceability.

### 5. Demo Interface (Streamlit)

- Simple and responsive interface (`app.py`)
- User sees:
  - Which tool/agent branch was used
  - Retrieved context (if any)
  - Final answer
- Port configuration for deployment (default: `8501`)

---

## File Structure

```plaintext
.
├── app.py                  # Streamlit frontend app
├── agent_rag_pipeline.py   # Core logic: agent, retrieval, and tools
├── download_model.py       # Downloads Mistral-7B model if missing
├── Documents/              # Folder containing .txt files for RAG
├── faiss_index/            # Saved vector index (auto-created)
├── mistral-7b-instruct...  # Local GGUF model (auto-downloaded)
├── nltk_data/              # WordNet data for dictionary tool
```

## Setup & Usage

### Prerequisites

- Python 3.10+
- Recommended: Run in a virtual environment
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App in the directory
```bash
streamlit run app.py
```
