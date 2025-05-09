import streamlit as st
import logging
from agent_rag_pipeline import get_vectorstore, process_query, retrieve_top_k_chunks
import nltk
nltk.data.path.append("/app/nltk_data")

# Setup
st.set_page_config(page_title="Smart QA Demo", layout="centered")
st.title("KSTech Smart Support Agent")

# Load Vectorstore
@st.cache_resource
def load_vs():
    return get_vectorstore()

vectorstore = load_vs()

# Input
query = st.text_input("Ask your question here:")

# Output
if query:
    # Detect tool
    if any(kw in query.lower() for kw in ["calculate", "compute"]):
        tool_used = "Calculator"
        context = "Not applicable."
    elif any(kw in query.lower() for kw in ["define", "meaning of", "what is the definition of"]):
        tool_used = "Dictionary"
        context = "Not applicable."
    else:
        tool_used = "RAG"
        docs = retrieve_top_k_chunks(vectorstore, query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Get answer
    with st.spinner("Thinking..."):
        answer = process_query(query, vectorstore)

    # Display results
    st.markdown(f"Tool/Agent Branch Used: `{tool_used}`")
    st.markdown("Retrieved Context:" if tool_used == "RAG" else "### 📚 Retrieved Context: Not applicable.")
    if tool_used == "RAG":
        st.text_area("Context Snippets", context, height=200)
    st.markdown("Final Answer:")
    st.success(answer)
