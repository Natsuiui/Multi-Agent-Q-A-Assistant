import os
import streamlit as st
from agent_rag_pipeline import get_vectorstore, process_query, retrieve_top_k_chunks

# For cloud deployment
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"

st.set_page_config(page_title="Smart QA Demo", layout="centered")
st.title("KSTech Smart Support Agent")

@st.cache_resource
def load_vs():
    return get_vectorstore()

vectorstore = load_vs()

query = st.text_input("Ask your question here:")

if query:
    query_lower = query.lower()

    MATH_KEYWORDS = [
        "plus", "minus", "times", "multiplied by", "divided", "square", "cube",
        "root", "factorial", "less than", "more than", "to the power of"
    ]
    DICT_KEYWORDS = ["define", "meaning of", "what is the definition of"]

    # Detect which tool to use
    if query_lower.startswith(("calculate", "compute")) or any(kw in query_lower for kw in MATH_KEYWORDS):
        tool_used = "Calculator"
    elif any(kw in query_lower for kw in DICT_KEYWORDS):
        tool_used = "Dictionary"
    else:
        tool_used = "RAG"

    # Get relevant context if using RAG
    if tool_used == "RAG":
        docs = retrieve_top_k_chunks(vectorstore, query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    else:
        context = None

    with st.spinner("Thinking..."):
        answer = process_query(query, vectorstore)

    st.markdown(f"**Tool/Agent Branch Used:** `{tool_used}`")

    if tool_used == "RAG":
        st.markdown("**Retrieved Context:**")
        st.text_area("Context Snippets", context, height=200)

    st.markdown("**Final Answer:**")
    st.success(answer)
