from flask import Flask, render_template, request
from agent_rag_pipeline import get_vectorstore, process_query, retrieve_top_k_chunks

# Initialize the Flask app
app = Flask(__name__)

# Load the vectorstore only once
vectorstore = get_vectorstore()

@app.route("/", methods=["GET", "POST"])
def home():
    tool_used = ""
    context = ""
    answer = ""
    
    if request.method == "POST":
        query = request.form["query"]
        
        if query:
            # Check for specific keywords and route accordingly
            if any(kw in query.lower() for kw in ["calculate", "compute"]):
                tool_used = "Calculator"
                context = "Not applicable."
                answer = process_query(query, vectorstore)
            elif any(kw in query.lower() for kw in ["define", "meaning of", "what is the definition of"]):
                tool_used = "Dictionary"
                context = "Not applicable."
                answer = process_query(query, vectorstore)
            else:
                tool_used = "RAG"
                docs = retrieve_top_k_chunks(vectorstore, query)
                context = "\n\n---\n\n".join([doc.page_content for doc in docs])
                answer = process_query(query, vectorstore)
    
    return render_template("index.html", tool_used=tool_used, context=context, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
