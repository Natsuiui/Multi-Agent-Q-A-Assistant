from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from pathlib import Path
import os

# Get the directory of the current script
base_dir = Path(__file__).resolve().parent

# Load text documents
def load_documents(file_paths):
    documents = []
    for path in file_paths:
        loader = TextLoader(path)
        documents.extend(loader.load())
    return documents

# Chunk documents
def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# Create FAISS index
def build_faiss_index(chunks):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    return vectorstore

# Retrieval function
def retrieve_top_k_chunks(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)


file_paths = [
    base_dir / "Documents" / "faqs.txt",
    base_dir / "Documents" / "smartphone_specs.txt",
    base_dir / "Documents" / "laptop_specs.txt",
    base_dir / "Documents" / "warranty_policy.txt",
    base_dir / "Documents" / "support_guide.txt"
]

docs = load_documents(file_paths)
chunks = chunk_documents(docs)
vector_index = build_faiss_index(chunks)

query = "What warranty comes with a smartphone?"
results = retrieve_top_k_chunks(vector_index, query)

print("Top 3 relevant chunks:\n")
for i, res in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(res.page_content, "\n")