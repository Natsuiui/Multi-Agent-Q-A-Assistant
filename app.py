import os
import streamlit as st
from agent_rag_pipeline import get_vectorstore, process_query, retrieve_top_k_chunks

# Define tool keywords
CALC_KEYWORDS = {"calculate", "compute", "plus", "minus", "times", "multiplied by", "divided", "square", "cube", "root", "factorial", "less", "more", "to the power of"}
DICT_KEYWORDS = {"define", "meaning of", "what is the definition of"}

def detect_tool(query):
    q = query.lower()
    if any(kw in q for kw in CALC_KEYWORDS):
        return "Calculator"
    elif any(kw in q for kw in DICT_KEYWORDS):
        return "Dictionary"
    else:
        return "RAG"

if __name__ == "__main__":
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"

    st.set_page_config(page_title="Smart QA Demo", layout="centered")
    st.title("KSTech Smart Support Agent")

    @st.cache_resource
    def load_vs():
        return get_vectorstore()

    vectorstore = load_vs()

    query = st.text_input("Ask your question here:").strip()

    if query:
        tool_used = detect_tool(query)

        if tool_used == "RAG":
            docs = retrieve_top_k_chunks(vectorstore, query)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        else:
            context = None

        with st.spinner("Thinking..."):
            try:
                answer = process_query(query, vectorstore)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                answer = "Error occurred while processing your query."

        st.markdown(f"**Tool/Agent Branch Used:** `{tool_used}`")

        if tool_used == "RAG":
            st.markdown("**Retrieved Context:**")
            st.text_area("Context Snippets", context, height=200)

        st.markdown("**Final Answer:**")
        st.success(answer)
