# %%
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

from dotenv import load_dotenv, find_dotenv
import os
# %%
# Load environment variables
load_dotenv('.env')
os.environ['OPENAI_API_KEY']
# %%

# Load PDF documents
def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

# Split documents into chunks
def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

# Load PDF file
file_name = './data/tesla10K.pdf'
data = load_documents(file_name)

# %%
# Chunk documents
texts = chunk_documents(data)
print(f'Now you have {len(texts)} documents')

# %%
## Set up embeddings and vector store
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(texts, embedding_function="gpt-turbo-3.5")
# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def chromadb_retrieval_qa(texts, question):
# Define the embedding function using SentenceTransformer
    embedding_function = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Use the embedding function with Chroma
    vectorstore = Chroma.from_documents(texts, embedding_function)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever()
    )
    result = qa_chain({"query": question})
    return result
# %%
if __name__ == '__main__':
    # question = "summarize the text?"
    question = "waht is tesla 2023 revenue"
    # result = qa_chain({"query": question})
    result = chromadb_retrieval_qa(texts, question)
    print(result["result"])


# %%
