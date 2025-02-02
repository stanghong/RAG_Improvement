# %%
# Standard and Third-Party Libraries
import os
import pickle
import requests
from dotenv import load_dotenv

# PDF Processing
from pypdf import PdfReader

# Text Splitting Utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Machine Learning and Natural Language Processing Libraries
from sentence_transformers import CrossEncoder
import numpy as np

# Load environment variables from a .env file
load_dotenv('.env')

# %%
def process_txt(filename):
    """
    Reads a PDF file, extracts its text from each page, and then splits the text into chunks.
    """
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    # Use a recursive splitter with various separators
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    return character_split_texts

# %%
def clean_text_list(text_list):
    """
    Cleans a list of text strings by replacing tab and newline characters,
    stripping whitespace, and recombining the text.
    """
    cleaned_texts = []
    for text in text_list:
        # Replace tab characters and newline characters with a single space
        text = text.replace('\t', ' ').replace('\n', ' ')
        # Split text into lines, remove empty lines, and strip extra spaces
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

# %%
def rank_doc(query=None, text_chunks=None, topN=5):
    """
    Uses a CrossEncoder to score and rank text chunks based on the query.
    Returns the top N ranked text chunks.
    """
    if query is None or text_chunks is None:
        print('Missing query or text chunks')
        return []
    # Initialize the CrossEncoder model
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Get scores for each (query, document) pair
    scores = reranker.predict([[query, doc] for doc in text_chunks])
    # Sort indices of scores in descending order
    top_indices = np.argsort(scores)[::-1][:topN]
    # Retrieve the top-ranked documents
    top_pairs = [text_chunks[index] for index in top_indices]
    return top_pairs

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
OLLAMA_MODEL = "deepseek-r1:latest"  # Use the locally installed DeepSeek model

# %%
def rag(pdf_file=None, query=None):
    """
    Performs a retrieval-augmented generation (RAG) operation by:
      1. Processing and cleaning text from the PDF.
      2. Ranking text chunks relative to the query.
      3. Using the Ollama API to generate an answer based on the top-ranked text.
    A cache is used to avoid reprocessing the PDF and ranking.
    """
    # Define cache directory and filenames
    cache_dir = './cache'
    base_filename = os.path.basename(pdf_file)
    cleaned_texts_file = os.path.join(cache_dir, f"{base_filename}_cleaned_texts.pickle")
    ranked_docs_file = os.path.join(cache_dir, f"{base_filename}_{query}_ranked_docs.pickle")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load cleaned texts from cache if available
    if os.path.exists(cleaned_texts_file):
        with open(cleaned_texts_file, 'rb') as f:
            print("Loading cleaned texts from cache.")
            cleaned_texts = pickle.load(f)
    else:
        # Process and clean texts if no cache is available
        character_split_texts = process_txt(pdf_file)
        cleaned_texts = clean_text_list(character_split_texts)
        # Cache the cleaned texts
        with open(cleaned_texts_file, 'wb') as f:
            pickle.dump(cleaned_texts, f)

    # Load ranked documents from cache if available
    if os.path.exists(ranked_docs_file):
        with open(ranked_docs_file, 'rb') as f:
            print("Loading ranked documents from cache.")
            retrieved_documents = pickle.load(f)
    else:
        # Rank the documents based on the query if no cache is available
        retrieved_documents = rank_doc(query, cleaned_texts, topN=5)
        # Cache the ranked documents
        with open(ranked_docs_file, 'wb') as f:
            pickle.dump(retrieved_documents, f)

    information = "\n\n".join(retrieved_documents)
    
    if query is None or not retrieved_documents:
        print('Missing query or retrieved documents')
        return None

    # Prepare the prompt for Ollama
    prompt = f"Question: {query}.\nInformation: {information}"

    # Call the Ollama API
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False  # Set to True if you want to stream the response
    }
    response = requests.post(OLLAMA_API_URL, json=data)
    
    if response.status_code == 200:
        content = response.json().get("response", "").strip()
        return content
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# %%
def process_folder(folder_path, query):
    """
    Processes all PDF files in the specified folder and applies the RAG process to each file.
    """
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in the folder: {folder_path}")
        return

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing file: {pdf_file}")
        output = rag(pdf_path, query)
        if output:
            print(f"Result for {pdf_file}:\n{output}\n")
        else:
            print(f"No result for {pdf_file}\n")

# %%
if __name__ == '__main__':
    query = "is google a good company to invest?"
    folder_path = "."  # Replace with the path to your folder containing PDFs
    process_folder(folder_path, query)
# %%
