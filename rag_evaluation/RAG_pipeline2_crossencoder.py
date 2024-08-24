# %%
# %%
# Standard and Third-Party Libraries
import os
import re
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Machine Learning and Natural Language Processing Libraries
from sentence_transformers import CrossEncoder
from sklearn.metrics import ndcg_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn import functional as F

# PDF Processing
from pypdf import PdfReader

# Text Splitting Utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# OpenAI and Utilities
import openai
from openai import OpenAI
from helper_utils import word_wrap

# Load environment variables
_ = load_dotenv('.env')


# %%
import pandas as pd
# %%
# model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
model_name = 'cross-encoder'
reader = PdfReader("./data/tesla10K.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

# Split text by sentences
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
# %%
# Function to clean and format each entry in the list
def clean_text_list(text_list):
    cleaned_texts = []
    for text in text_list:
        # Replace tab characters with a single space
        text = text.replace('\t', ' ')
        text = text.replace('\n', ' ')
        # Split text into lines and remove any leading/trailing whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Combine lines back into a single string with newline characters
        cleaned_text = '\n'.join(lines)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

# Applying the function to the original list
formatted_texts = clean_text_list(character_split_texts)
# len(formatted_texts)

# %%
# text_chunks=df_qa1['doc']
text_chunks=formatted_texts


def rank_doc(query=None, text_chunks=None, topN=5):
    # Initialize the CrossEncoder model with the specified model name
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    if query is None or text_chunks is None:
        print('missing query or text chunk')

    # Predict scores for each document in relation to the query
    scores = reranker.predict([[query, doc] for doc in text_chunks])

    # Get indices of the top N scores in descending order
    top_indices = np.argsort(scores)[::-1][:topN]

    # Retrieve the top-ranked text documents using list indexing
    top_pairs = [text_chunks[index] for index in top_indices]
    return top_pairs  # Returns a list of the top-ranked text strings
# %%
# retrieved_documents = rank_doc(query, text_chunks)
# retrieved_documents
# %%

openai.api_key = os.environ['OPENAI_API_KEY']
openai_client = OpenAI()
# %%
def rag(query=None, retrieved_documents=None, model="gpt-3.5-turbo"):
    information = "\n\n".join(retrieved_documents)

    if query is None or retrieved_documents is None:
        print('missing query or retrieved documents')

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual 10K report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

# %%
if __name__ == '__main__':
    query = "what is revenue for 2023?"
    retrieved_documents = rank_doc(query, text_chunks)
    output = rag(query=query, retrieved_documents=retrieved_documents)
    print(word_wrap(output))

# %%
