# %%
# 1. load ragas
from helper_utils import word_wrap, convert_texts_to_documents
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder

import pandas as pd
from RAG_pipeline1_chromadb import chromadb_retrieval_qa
from RAG_pipeline2_crossencoder import rag, rank_doc

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from ragas import evaluate
from datasets import Dataset

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

# %%

_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']



# # Load Tesla 2023 10K report
pdf_file='./data/tesla10K.pdf'


# %%
# preprocess pdf into clean texts
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


# %%
text_list=process_pdf_texts(pdf_file)
#create clean texts chunks
cleaned_texts = clean_text_list(text_list)


def evaluate_ragas_dataset(ragas_dataset):
    result = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            context_relevancy,
            answer_correctness,
            answer_similarity
        ],
    )
    return result

# %%
def create_ragas_dataset(rag_pipeline, eval_dataset):
    rag_dataset = []
    for row in tqdm(eval_dataset):
        try:
            # Assuming rag_pipeline is a callable function that accepts a question
            answer = rag_pipeline(retrieved_documents, row["question"]) # Update based on your actual pipeline usage

            content = "No content available"
            contexts = ["No detailed context available"]

            # Check the type and contents of the answer
            if isinstance(answer, dict):
                content = answer.get('result', content)  # Safely get result from answer
                # Ensure 'context' is in answer and is a list before extracting
                if 'context' in answer and isinstance(answer['context'], list):
                    contexts = [context.page_content for context in answer['context']]

            elif isinstance(answer, str):
                # If answer is a string, directly use it as content
                content = answer

            # Append the collected data to rag_dataset
            rag_dataset.append({
                "question": row["question"],
                "answer": content,
                "contexts": [row['context']],
                "ground_truths": [row["ground_truth"]]
            })
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            continue  # Optionally skip to next row or handle error differently

    # Convert the list of dictionaries to a DataFrame and then to an Arrow Dataset
    rag_df = pd.DataFrame(rag_dataset)
    rag_eval_dataset = Dataset.from_pandas(rag_df)
    return rag_eval_dataset

# %%
eval_dataset = Dataset.from_csv("./data/groundtruth_eval_dataset.csv")


# %%
# Create the RAGAS dataset
ragas_dataset_pline1 = create_ragas_dataset(chromadb_retrieval_qa, eval_dataset)


# %%
evaluation_results_pline1 = evaluate_ragas_dataset(ragas_dataset_pline1)


# %%
df_pl1 = ragas_dataset_pline1.to_pandas()
# df.to_excel('qc_metrics_pline1.xlsx')


# %%
# query = "what is revenue for 2023?"
# retrieved_documents = rank_doc(cleaned_texts, query=query)
# output = rag(query=query, retrieved_documents=retrieved_documents)


# %%
# Create the RAGAS dataset
ragas_dataset_pline2 = create_ragas_dataset(rag, eval_dataset)

# %%
evaluation_results_pline2 = evaluate_ragas_dataset(ragas_dataset_pline2)
# %%
df_pl2 = ragas_dataset_pline2.to_pandas()
df_pl2.to_excel('qc_metrics_pline2.xlsx')
# evaluation_results_pipeline2 = evaluate_ragas_dataset(ragas_dataset)
# %%

result = pd.concat([df_pl2, df_pl2], axis=1)
result.to_excel('merged.xlsx')

# %%
# Create a DataFrame
df = pd.DataFrame({'ChromaDB ': evaluation_results_pline1, \
                   'ReRanker': evaluation_results_pline2})
df_transposed = df.transpose()


df_transposed
# %%
