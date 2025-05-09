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
from llama_cpp import Llama

# Logging and warning config
os.environ["GGML_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# NLTK path setup
nltk.data.path.append(str(Path(__file__).resolve().parent / "nltk_data"))

# Directories
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "Documents"
INDEX_DIR = BASE_DIR / "faiss_index"
MODEL_NAME = "mistral-7b-instruct-v0.1.Q2_K.gguf"

# Extract answer

def extract_answer(text):
    match = re.search(r'Answer:.*', text, re.DOTALL)
    return match.group(0) if match else text

# Calculator tool

def mock_calculator(query):
    try:
        expr = query.lower().replace("calculate", "").replace("compute", "").strip()
        result = eval(expr)
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

# Tools list
tools = [
    Tool(name="Calculator", func=mock_calculator, description="Performs calculations"),
    Tool(name="Dictionary", func=mock_dictionary, description="Defines a term or acronym"),
]

# Cached LLM loading
_llm_model = None

def get_llm():
    global _llm_model
    if _llm_model is None:
        _llm_model = Llama.from_pretrained(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            filename=MODEL_NAME,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=32
        )
    return _llm_model

# Get LLM response

def get_llm_response(prompt):
    model = get_llm()
    output = model(prompt, max_tokens=256, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# Load documents

def load_documents(file_paths):
    documents = []
    for path in file_paths:
        loader = TextLoader(path)
        documents.extend(loader.load())
    return documents

# Chunk and embed

def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_faiss_index(chunks):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding)


def retrieve_top_k_chunks(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)

# Load or build index

def get_vectorstore():
    if INDEX_DIR.exists():
        return FAISS.load_local(INDEX_DIR, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    else:
        files = [DOCS_DIR / fname for fname in os.listdir(DOCS_DIR) if fname.endswith(".txt")]
        docs = load_documents(files)
        chunks = chunk_documents(docs)
        vs = build_faiss_index(chunks)
        vs.save_local(INDEX_DIR)
        return vs

# LangChain agent (not used in app.py but reusable)

def get_langchain_agent():
    from langchain.chat_models import ChatOpenAI
    dummy_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    return initialize_agent(tools=tools, llm=dummy_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Main processor

def process_query(query, vectorstore):
    logging.info(f"Received query: {query}")
    if any(kw in query.lower() for kw in ["calculate", "compute"]):
        return mock_calculator(query)
    elif any(kw in query.lower() for kw in ["define", "meaning of", "what is the definition of"]):
        return mock_dictionary(query)
    else:
        docs = retrieve_top_k_chunks(vectorstore, query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"""
        raw = get_llm_response(prompt)
        return extract_answer(raw)