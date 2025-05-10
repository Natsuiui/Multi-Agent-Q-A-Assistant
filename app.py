import os
import streamlit as st
from agent_rag_pipeline import get_vectorstore, process_query, retrieve_top_k_chunks

if __name__ == "__main__":
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"  # For cloud deployment
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"        # Default local port

    st.set_page_config(page_title="Smart QA Demo", layout="centered")
    st.title("KSTech Smart Support Agent")

    @st.cache_resource
    def load_vs():
        return get_vectorstore()

    vectorstore = load_vs()

    query = st.text_input("Ask your question here:")

    if query:
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["calculate", "compute", "plus", "minus", "times", "divided", "square", "cube", "root", "factorial", "less", "more", "to the power of"]):
            tool_used = "Calculator"
        elif any(kw in query_lower for kw in ["define", "meaning of", "what is the definition of"]):
            tool_used = "Dictionary"
        else:
            tool_used = "RAG"

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
