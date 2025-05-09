# Smart QA Support Agent

This project is a **retrieval-augmented generation (RAG)** based question-answering system designed to assist users with technical support questions. It dynamically chooses between a **Calculator**, a **Dictionary**, and a **Contextual RAG agent** based on the nature of the query.

---

## Architecture Overview

- **Hybrid Agent System**: Uses LangChain's `ZERO_SHOT_REACT_DESCRIPTION` agent to orchestrate tool routing.
- **Local LLM**: Uses the `llama.cpp` integration with a quantized Mistral 7B model (`GGUF` format) for contextual answers.
- **Tools**: 
  - **Calculator**: Evaluates mathematical expressions.
  - **Dictionary**: Uses WordNet to provide word definitions.
  - **RAG Pipeline**:
    - Loads support documents (`faqs.txt`, `support_guide.txt`, etc.).
    - Chunks and indexes them using FAISS and SentenceTransformer embeddings.
    - Retrieves top-K relevant chunks per query.
    - Prompts the LLM using retrieved context.

---

## Key Design Choices

- **Routing Logic**: Queries are routed based on keyword detection to avoid unnecessary LLM calls.
- **LLM Efficiency**: Heavy LLM inference is reserved only for open-ended technical questions.
- **Persistence**: FAISS index is saved and reused to avoid repeated computation.
- **Streamlit UI**: Provides a simple front-end for demo and testing.

---

## How to Run

1. Run in Terminal:
```bash
  python agent_rag_pipeline.py
```

2. Run the Web App using streamlit (Recommended):
```bash
streamlit run app.py
```
3. Run the Web App using Flask:
- run `flask_app.py`.
- go to `http://127.0.0.1:5000` on your web browser.
