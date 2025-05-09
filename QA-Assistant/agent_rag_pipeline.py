import logging
import os
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from llama_cpp import Llama
from nltk.corpus import wordnet
import re

import warnings
from pathlib import Path
import logging

os.environ["GGML_LOG_LEVEL"] = "ERROR"

# Suppress all warnings
warnings.filterwarnings("ignore")

# Logging Setup
logging.basicConfig(level=logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Base Directory
base_dir = Path(__file__).resolve().parent / "Documents"

# Answer Formatting Setup
def extract_answer(text):
    match = re.search(r'Answer:.*', text, re.DOTALL)
    return match.group(0) if match else text

# Tools (Dictionary and Calculator)
def mock_calculator(query: str) -> str:
    try:
        expr = query.lower().replace("calculate", "").replace("compute", "").strip()
        result = eval(expr)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

def mock_dictionary(query: str) -> str:
    word = query.strip().split()[-1]
    synsets = wordnet.synsets(word)

    if not synsets:
        return f"No definition found for '{word}'."

    definitions = [f"{i+1}. ({syn.pos()}) {syn.definition()}" for i, syn in enumerate(synsets)]
    return "\n".join(definitions[:5])

tools = [
    Tool(name="Calculator", func=mock_calculator, description="Performs calculations"),
    Tool(name="Dictionary", func=mock_dictionary, description="Defines a term or acronym"),
]

# LLM Wrapper
def get_llm_response(prompt: str) -> str:
    model = Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        filename="mistral-7b-instruct-v0.1.Q2_K.gguf",
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=32
    )
    output = model(prompt, max_tokens=256, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# RAG Functions
def load_documents(file_paths):
    documents = []
    for path in file_paths:
        loader = TextLoader(path)
        documents.extend(loader.load())
    return documents

def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def build_faiss_index(chunks):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding)

def retrieve_top_k_chunks(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)

# Load or Build VectorStore
def get_vectorstore():
    if os.path.exists("QA-Assistant/faiss_index"):
        return FAISS.load_local("QA-Assistant/faiss_index", SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    else:
        file_paths = [
            str(base_dir / "faqs.txt"),
            str(base_dir / "laptop_specs.txt"),
            str(base_dir / "smartphone_specs.txt"),
            str(base_dir / "support_guide.txt"),
            str(base_dir / "warranty_policy.txt")
            ]
        docs = load_documents(file_paths)
        chunks = chunk_documents(docs)
        vs = build_faiss_index(chunks)
        vs.save_local("QA-Assistant/faiss_index")
        return vs

# Agent Initialization
def get_langchain_agent():
    from langchain.chat_models import ChatOpenAI
    dummy_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # Only used by the agent router
    return initialize_agent(tools=tools, llm=dummy_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Master Query Processor
def process_query(query: str, vectorstore) -> str:
    logging.info(f"Received query: {query}")

    if any(kw in query.lower() for kw in ["calculate", "compute", "define", "meaning of", "what is the definition of"]):
        if "calculate" in query.lower() or "compute" in query.lower():
            logging.info("Routing to Calculator Tool")
            return mock_calculator(query)
        else:
            logging.info("Routing to Dictionary Tool")
            return mock_dictionary(query)
    else:
        logging.info("Routing to RAG Pipeline")
        docs = retrieve_top_k_chunks(vectorstore, query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"""
        raw_response = get_llm_response(prompt)
        return extract_answer(raw_response)

# Run Loop
if __name__ == "__main__":
    vs = get_vectorstore()

    print("Ask your technical support question (type 'exit' to quit):")
    while True:
        query = input("\n> ")
        if query.lower() == 'exit':
            break
        result = process_query(query, vs)
        print(f"\n\n{result}")