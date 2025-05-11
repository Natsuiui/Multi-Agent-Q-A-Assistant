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

# Logging and warnings
os.environ["GGML_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# NLTK path and data download
BASE_DIR = Path(__file__).resolve().parent
nltk.data.path.append(str(BASE_DIR / "nltk_data"))
nltk.download('wordnet', download_dir=str(BASE_DIR / "nltk_data"), quiet=True)

# Directories and model path
DOCS_DIR = BASE_DIR / "Documents"
INDEX_DIR = BASE_DIR / "faiss_index"
MODEL_NAME = str(BASE_DIR / "mistral-7b-instruct-v0.1.Q2_K.gguf")

# Extract answer
def extract_answer(text):
    match = re.search(r'Answer:.*', text, re.DOTALL)
    return match.group(0) if match else text

# Calculator tool
import sympy
from sympy import factorial

# Replacements for natural language math terms
NATURAL_MATH_REPLACEMENTS = {
    r"\bplus\b": "+",
    r"\bminus\b": "-",
    r"\btimes\b|\bmultiplied by\b": "*",
    r"\bdivided by\b": "/",
    r"\bsquared\b|\bsquare\b": "**2",
    r"\bcubed\b": "**3",
    r"\bfactorial of\b": "factorial",
    r"\bto the power of (\d+)\b": r"**\1",
    r"\bless than\b": "<",
    r"\bmore than\b": ">",
}

def wrap_functions(expr):
    expr = re.sub(r'(\d+)!', r'factorial(\1)', expr)  # Handle 5! → factorial(5)
    expr = re.sub(r'\bfactorial\s*\(?\s*(\d+)\s*\)?', r'factorial(\1)', expr)  # normalize factorial usage
    return expr

# Calculator function
def mock_calculator(query):
    try:
        expr = query.lower().strip()

        # Remove 'calculate' or 'compute' prefix
        expr = re.sub(r"^(calculate|compute)\s*", "", expr)

        # Handle root phrases (e.g., square root of 25 → (25**0.5))
        expr = re.sub(r"square root of\s*(\d+(\.\d+)?)", r"(\1**0.5)", expr)
        expr = re.sub(r"cube root of\s*(\d+(\.\d+)?)", r"(\1**(1/3))", expr)

        # Only replace natural-language patterns
        for pattern, replacement in NATURAL_MATH_REPLACEMENTS.items():
            expr = re.sub(pattern, replacement, expr)

        # Wrap special functions
        expr = wrap_functions(expr)

        # Evaluate the expression
        result = sympy.sympify(expr, evaluate=True)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# Dictionary tool
def mock_dictionary(query):
    word = query.strip().split()[-1]
    synsets = wordnet.synsets(word)
    if not synsets:
        return f"No definition found for '{word}'."
    definitions = [f"{i+1}. ({syn.pos()}) {syn.definition()}" for i, syn in enumerate(synsets)]
    return "\n".join(definitions[:5])

# Tool list
tools = [
    Tool(name="Calculator", func=mock_calculator, description="Performs calculations"),
    Tool(name="Dictionary", func=mock_dictionary, description="Defines a term or acronym"),
]

# Load local GGUF LLM model (Commented Out)
"""
_llm_model = None

def get_llm():
    global _llm_model
    if _llm_model is None:
        _llm_model = Llama(
            model_path=MODEL_NAME,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=32
        )
    return _llm_model
"""

import requests

LLM_SERVER_URL = "https://a340-2405-201-4018-2c04-20fa-2b60-b0ec-2c.ngrok-free.app/generate"

def get_llm_response(prompt):
    try:
        response = requests.post(LLM_SERVER_URL, json={"prompt": prompt}, timeout=20)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException:
        return "LLM is currently offline. Please contact kshitijsharma1106@gmail.com, or call +91 8448927161"
    except Exception as e:
        return f"Unexpected error: {e}"

# Load and chunk documents
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

# Load or build FAISS index
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

# LangChain agent (optional use)
def get_langchain_agent():
    from langchain.chat_models import ChatOpenAI
    dummy_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    return initialize_agent(tools=tools, llm=dummy_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Query processor
def process_query(query, vectorstore):
    logging.info(f"Received query: {query}")
    math_keywords = [
        "calculate", "compute", "plus", "minus", "times", "multiplied", "divided",
        "square", "cube", "root", "factorial", "less than", "more", "to the power of"
    ]
    if any(kw in query.lower() for kw in math_keywords):
        return mock_calculator(query)
    elif any(kw in query.lower() for kw in ["define", "meaning of", "what is the definition of"]):
        return mock_dictionary(query)
    else:
        docs = retrieve_top_k_chunks(vectorstore, query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"""
        raw = get_llm_response(prompt)
        return extract_answer(raw)
