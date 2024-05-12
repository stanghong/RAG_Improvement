# %%
from helper_utils import word_wrap
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import umap.umap_ as umap
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder
_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']
openai_client = OpenAI()

# %%
# Load the 2023 Tesla 10K report
NeedRefreshVDB=False
if NeedRefreshVDB:
    reader = PdfReader("../data/tesla10K.pdf")
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

    # Split text by sentences
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    # Tokenize the sentence chunks
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = [token_split_text for text in character_split_texts for token_split_text in token_splitter.split_text(text)]

    # Create embeddings
    embedding_function = SentenceTransformerEmbeddingFunction()

    # Create vector database using ChromaDB collection
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("tesla202310k", embedding_function=embedding_function)
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)
else:
    # Load existing vector database # improvement 1
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.Client()
    vectordb = chroma_client.get_collection("tesla202310k",
    embedding_function=embedding_function)

# Function to calculate NDCG
def ndcg(scores, ideal_scores, k):
    dcg = lambda scores: sum((2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores[:k]))
    actual_dcg = dcg(scores)
    ideal_dcg = dcg(sorted(ideal_scores, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

# %%
def rag(query, retrieved_documents, model="gpt-3.5-turbo"):
    information = "\n\n".join(retrieved_documents)

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

query = "What was the total revenue?"
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results['documents'][0]
output = rag(query=query, retrieved_documents=retrieved_documents)
print(word_wrap(output))

# %%
# -----------------create pairs -----------------------
def create_query_retrival_pairs(query): 
    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]

    pairs = []
    for doc in retrieved_documents:
        pairs.append([query, doc])

    return pairs, retrieved_documents

query = 'what is 2023 revenue for tesla'
pairs, retrieved_documents= create_query_retrival_pairs(query)
# %%
from sentence_transformers import CrossEncoder

def rank_relevancy_pairs(pairs):
    topN=5
    # Initialize the CrossEncoder with a specific pre-trained model
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Predict scores for the pairs using the CrossEncoder
    scores = cross_encoder.predict(pairs)
    
    # Select top 5 ranked answers by sorting the indices of scores in descending order
    top_indices = np.argsort(scores)[::-1][:topN]  # Select only the top 5
    
    # Assign relevancy scores from 4 to 0 (or less based on list length)
    predicted_relevance = [0] * topN
    for rank, index in enumerate(top_indices):
        predicted_relevance[index] = topN-1 - rank  # Adjust this based on how many scores there are
    
    # Retrieve the top 5 pairs based on these indices
    top_pairs = [pairs[index] for index in top_indices]
    
    return top_indices, predicted_relevance, top_pairs
# %%

def true_relevancy(pairs, model="gpt-4"):

    relevancy_scores = [0]*5

    # for question, answer in pairs:
    messages = [
    {
        "role": "system",
        "content": (
        "You are a helpful expert text processing expert.\n"
        "- Return relevancy ranking for each of the provided pairs in a simple list format"
        "- each value corresponds to the relevancy score between 0 and 4 for each pair in the order they were presented.\n"
        "- Rate the relevancy of the answer to the question on a scale of 0 to 4, where 4 is highly relevant and 0 is not relevant at all.\n"
        "- if there is relevant results, there cannot be duplicated relevancy score \n"
        "- return answer into a list no explaination\n"
        "- Answer the user's question using only this information."
    )
    },
    {"role": "user", "content": f"Question and answer pairs are {pairs}. \n "}

    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract the relevancy score from the response
    try:
        # Assuming the response is a single digit or a simple number
        
        relevancy_scores=response.choices[0].message.content
        print(f'evaluated by chatGPT4: relevancy score is:{relevancy_scores}')
    except Exception as e:
        print(f"Error parsing response: {e}")
        # Append a default score or handle error appropriately
        relevancy_scores=[0,0,0,0,0]

    # handle non relevant texts 
    # if multiple texts are found zeros, return all zeros
    if isinstance(relevancy_scores, str):
        true_relevance_score = eval(relevancy_scores)  
    else: 
        true_relevance_score = relevancy_scores
    
    true_relevance_score = [0, 0, 0, 0, 0] if true_relevance_score.count(0) > 1 else true_relevance_score

    return true_relevance_score 
# %%
# -------------------------------------------------------------------
query = "What was linonel messi contribution?"
pairs, retrieved_documents= create_query_retrival_pairs(query)
output = rag(query=query, retrieved_documents=retrieved_documents)
print(word_wrap(output))

# %%
top_indices, predicted_relevance, top_pairs = rank_relevancy_pairs(pairs)

print("Predicted Relevancy Scores:", predicted_relevance)

# %%
# GPT 4 evaluated relevancy as ground truth
true_relevance_score  = true_relevancy(pairs)

# %%
# # Calculate NDCG score
ndcg_score = ndcg(predicted_relevance, true_relevance_score, k=5)
print(f"NDCG Score: {ndcg_score}")
# %%
# ---------------------------------------------------
query = "What was total revenue of 2023?"
pairs, retrieved_documents = create_query_retrival_pairs(query)

output = rag(query=query, retrieved_documents=retrieved_documents)
print(word_wrap(output))

# %%
top_indices, predicted_relevance, top_pairs = rank_relevancy_pairs(pairs)
print("Predicted Relevancy Scores:", predicted_relevance)

# %%
# GPT 4 evaluated relevancy as ground truth
true_relevance_score = true_relevancy(pairs)
# %%
# # Calculate NDCG score
ndcg_score = ndcg(predicted_relevance, true_relevance_score, k=5)
print(f"NDCG Score: {ndcg_score}")
# %%
# ---------------------------------------------------
query = "What was total revenue of 2022?"
pairs, retrieved_documents = create_query_retrival_pairs(query)
output = rag(query=query, retrieved_documents=retrieved_documents)
print(word_wrap(output))

# %%
top_indices, predicted_relevance, top_pairs = rank_relevancy_pairs(pairs)
print("Predicted Relevancy Scores:", predicted_relevance)

# %%
# GPT 4 evaluated relevancy as ground truth
true_relevance_score = true_relevancy(pairs)
# %%
# # Calculate NDCG score
ndcg_score = ndcg(predicted_relevance, true_relevance_score, k=5)
print(f"NDCG Score: {ndcg_score}")

# %%
# ---------------------------------------------------
query = "Is Elon Musk genius or not?"
pairs, retrieved_documents = create_query_retrival_pairs(query)
output = rag(query=query, retrieved_documents=retrieved_documents)
print(word_wrap(output))

# %%
top_indices, predicted_relevance, top_pairs = rank_relevancy_pairs(pairs)
print("Predicted Relevancy Scores:", predicted_relevance)
# %%
# GPT 4 evaluated relevancy as ground truth
true_relevance_score = true_relevancy(pairs)


# # Calculate NDCG score
ndcg_score = ndcg(predicted_relevance, true_relevance_score, k=5)
print(f"NDCG Score: {ndcg_score}")
# %%
##TODO: GPT$ evaluation is not stable need to improve code

