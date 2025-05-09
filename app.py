import os
import streamlit as st
from agent_rag_pipeline import get_vectorstore, process_query, retrieve_top_k_chunks

from download_model import download_model
download_model()  # Will skip if already downloaded

# Set Streamlit config for external access (adjust if local testing)
if __name__ == "__main__":
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"  # For cloud
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"        # Use 8501 for local testing

    st.set_page_config(page_title="Smart QA Demo", layout="centered")
    st.title("KSTech Smart Support Agent")

    @st.cache_resource
    def load_vs():
        return get_vectorstore()

    vectorstore = load_vs()

    query = st.text_input("Ask your question here:")

    if query:
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

        with st.spinner("Thinking..."):
            answer = process_query(query, vectorstore)

        st.markdown(f"Tool/Agent Branch Used: `{tool_used}`")
        if tool_used == "RAG":
            st.markdown("Retrieved Context:")
            st.text_area("Context Snippets", context, height=200)
        else:
            st.markdown("Retrieved Context: Not applicable.")

        st.markdown("Final Answer:")
        st.success(answer)
