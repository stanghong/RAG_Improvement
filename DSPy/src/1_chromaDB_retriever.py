
## code to turn pdf into collection can be used by DSPy

import os
import re
from pypdf import PdfReader
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

load_dotenv()  # Load environment variables

# Load and process PDF documents
def load_documents(file_path):
    reader = PdfReader(file_path)
    return [p.extract_text().strip() for p in reader.pages if p.extract_text()]

def clean_text(text):
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def chunk_text(text, size=1000):
    """Split text into chunks of approximately 'size' characters."""
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0
    for word in words:
        if current_length + len(word) + 1 > size:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_length = [word], len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def process_pdf(file_path):
    """Load, clean, and chunk text from a PDF."""
    pdf_texts = load_documents(file_path)
    full_text = '\n'.join(pdf_texts)
    cleaned_text = clean_text(full_text)
    return chunk_text(cleaned_text)

# Load PDF file and prepare text
pdf_file = '../data/tesla10K.pdf'
token_split_texts = process_pdf(pdf_file)

# Initialize ChromaDB and add documents
chroma_client = chromadb.PersistentClient(path="./teslasec")
collection = chroma_client.get_or_create_collection(
    name="tesla",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)
collection.add(documents=token_split_texts, ids=[str(i) for i in range(len(token_split_texts))])

# Set up DSPy retriever model
llm = dspy.OpenAI(model="gpt-3.5-turbo")
retriever_model = ChromadbRM(
    collection_name='tesla', 
    persist_directory="./teslasec",
    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    k=5
)

dspy.settings.configure(lm=llm, rm=retriever_model)

# Example query to test the retriever
retriever_model("revenue")

