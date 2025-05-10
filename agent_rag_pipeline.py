import os
import re
import logging
import warnings
from pathlib import Path
import nltk
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from nltk.corpus import wordnet
import requests
import sympy
from sympy import factorial, sqrt

# Logging and warnings
os.environ["GGML_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# NLTK setup
BASE_DIR = Path(__file__).resolve().parent
nltk.data.path.append(str(BASE_DIR / "nltk_data"))
nltk.download('wordnet', download_dir=str(BASE_DIR / "nltk_data"), quiet=True)

# Paths
DOCS_DIR = BASE_DIR / "Documents"
INDEX_DIR = BASE_DIR / "faiss_index"
MODEL_NAME = str(BASE_DIR / "mistral-7b-instruct-v0.1.Q2_K.gguf")

# Natural language to math symbol replacements
NATURAL_MATH_REPLACEMENTS = {
    r"\bplus\b": "+",
    r"\bminus\b": "-",
    r"\btimes\b|\bmultiplied by\b": "*",
    r"\bdivided by\b": "/",
    r"\bsquared\b": "**2",
    r"\bcubed\b": "**3",
    r"\bsquare root of\b": "sqrt ",
    r"\bcube root of\b": "cbrt ",
    r"\bfactorial of\b": "factorial ",
    r"\bto the power of (\d+)\b": r"**\1",
    r"\bless than\b": "<",
    r"\bmore than\b": ">",
}

# Calculator Tool
def mock_calculator(query):
    try:
        expr = query.lower()
        expr = re.sub(r"(calculate|compute|what is|the|of|value|result)", "", expr)

        for pattern, replacement in NATURAL_MATH_REPLACEMENTS.items():
            expr = re.sub(pattern, replacement, expr)

        expr = re.sub(r'cbrt\s*(\d+(\.\d+)?)', r'(\1)**(1/3)', expr)
        expr = re.sub(r'sqrt\s*(\d+(\.\d+)?)', r'sqrt(\1)', expr)
        expr = re.sub(r'factorial\s*(\d+)', r'factorial(\1)', expr)

        expr = expr.strip()
        result = sympy.sympify(expr, evaluate=True)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# Dictionary Tool
def mock_dictionary(query):
    word = query.strip().split()[-1]
    synsets = wordnet.synsets(word)
    if not synsets:
        return f"No definition found for '{word}'."
    definitions = [f"{i+1}. ({syn.pos()}) {syn.definition()}" for i, syn in enumerate(synsets)]
    return "\n".join(definitions[:5])

# LLM Response via API
LLM_SERVER_URL = "https://0fc9-2405-201-4018-2c04-d48e-b2da-199a-f4bf.ngrok-free.app/generate"

def get_llm_response(prompt):
    try:
        response = requests.post(LLM_SERVER_URL, json={"prompt": prompt}, timeout=20)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException:
        return "LLM is currently offline. Please contact kshitijsharma1106@gmail.com"
    except Exception as e:
        return f"Unexpected error: {e}"

# Helper: Extract answer from response
def extract_answer(text):
    match = re.search(r'Answer:.*', text, re.DOTALL)
    return match.group(0) if match else text

# FAISS-related logic
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

def get_vectorstore():
    if INDEX_DIR.exists():
        return FAISS.load_local(INDEX_DIR, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    else:
        files = [DOCS_DIR / fname for fname in os.listdir(DOCS_DIR) if fname.endswith(".txt")]
        if not files:
            raise RuntimeError(f"No .txt files found in {DOCS_DIR}")
        docs = load_documents(files)
        chunks = chunk_documents(docs)
        vs = build_faiss_index(chunks)
        vs.save_local(INDEX_DIR)
        return vs

# Agent (optional use)
def get_langchain_agent():
    from langchain.chat_models import ChatOpenAI
    dummy_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    return initialize_agent(tools=[
        Tool(name="Calculator", func=mock_calculator, description="Performs calculations"),
        Tool(name="Dictionary", func=mock_dictionary, description="Defines a term or acronym"),
    ], llm=dummy_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Query Processor (UPDATED)
def process_query(query, vectorstore):
    logging.info(f"Received query: {query}")
    math_keywords = ["calculate", "compute", "plus", "minus", "times", "multiplied", "divided", "square", "cube", "root", "factorial", "less than", "more", "to the power of"]
    dict_keywords = ["define", "meaning of", "what is the definition of"]

    lower_query = query.lower()
    if any(kw in lower_query for kw in math_keywords):
        return mock_calculator(query)
    elif any(kw in lower_query for kw in dict_keywords):
        return mock_dictionary(query)
    else:
        docs = retrieve_top_k_chunks(vectorstore, query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"""
        raw = get_llm_response(prompt)
        return extract_answer(raw)
