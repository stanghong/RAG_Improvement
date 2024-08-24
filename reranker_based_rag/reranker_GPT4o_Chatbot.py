
from openai import OpenAI
# Text Splitting Utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import CrossEncoder

import streamlit as st
from pypdf import PdfReader
import openai
import numpy as np

st.set_page_config(page_title="PDF Query Assistant with Reranker and Completely Free", layout="wide")

def rank_doc(query, text_chunks, topN=5):
    # Initialize the CrossEncoder model with the specified model name
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Predict scores for each document in relation to the query
    scores = reranker.predict([[query, doc] for doc in text_chunks])

    # Get indices of the top N scores in descending order
    top_indices = np.argsort(scores)[::-1][:topN]

    # Retrieve the top-ranked text documents using list indexing
    top_pairs = [text_chunks[index] for index in top_indices]
    return top_pairs  # Returns a list of the top-ranked text strings


def rag(query, retrieved_documents, api_key):
    model = "gpt-4o"


    # Set the API key
    openai.api_key = api_key

    information = "\n\n".join(retrieved_documents)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual 10K report."
                       "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content # Updated to correct attribute access
    return content


@st.cache_data
def process_pdf_texts(pdf_file):
    reader = PdfReader(pdf_file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    return clean_text_list(character_split_texts)

def clean_text_list(text_list):
    cleaned_texts = []
    for text in text_list:
        text = text.replace('\t', ' ').replace('\n', ' ')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=['pdf'])

if uploaded_file and api_key:
    formatted_texts = process_pdf_texts(uploaded_file)
    st.session_state.processed_texts = formatted_texts

st.title("Free PDF Query Assistant with Reranker")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_history:
    for query, response in st.session_state.chat_history:
        st.container().markdown(f"**Q**: {query}")
        st.container().markdown(f"**A**: {response}")

query = st.text_input("Type your question here:", key="query")

if st.button("Submit Query"):
    if 'processed_texts' in st.session_state and query and api_key:
        with st.spinner('Processing...'):
            retrieved_documents = rank_doc(query, st.session_state.processed_texts)
            output_wrapped = rag(query, retrieved_documents, api_key)
            st.session_state.chat_history.append((query, output_wrapped))
            st.container().markdown(f"**Q**: {query}")
            st.container().markdown(f"**A**: {output_wrapped}")
    else:
        st.error("Please upload a PDF, ensure the API key is set, and type a question.")
