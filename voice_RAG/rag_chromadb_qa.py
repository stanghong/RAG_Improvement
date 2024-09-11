# %%
# File: rag_chromadb.py
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
# %%
# Function to load PDF documents
def load_documents(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()
# %%
# Function to split documents into chunks
def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)
# %%
# Function to run Chroma-based retrieval and QA
def chromadb_retrieval_qa(texts, question):
    # embedding_function = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    

    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore = Chroma.from_documents(texts, embedding_function)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever()
    )
    result = qa_chain({"query": question})
    return result["result"]
# %%
# If you want to run the script directly
if __name__ == '__main__':
    file_name = './data/Self-correcting LLM-controlled Diffusion Models.pdf'
    question = "What is LLM?"

    data = load_documents(file_name)
    texts = chunk_documents(data)

    result = chromadb_retrieval_qa(texts, question)
    print(result)

# %%
