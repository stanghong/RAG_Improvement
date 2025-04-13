from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

# Load PDF documents
def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

# Split documents into chunks
def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

# Load PDF file
file_name = 'tesla10K.pdf'
data = load_documents(file_name)

# Chunk documents
texts = chunk_documents(data)
print(f'Now you have {len(texts)} documents')

def chromadb_retrieval_qa(texts, question):
    # Define the embedding function using SentenceTransformer
    embedding_function = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Use the embedding function with Chroma
    vectorstore = Chroma.from_documents(
        texts, 
        embedding_function, 
        persist_directory="./financial_analysis_db"
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever()
    )
    result = qa_chain({"query": question})
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python rag.py <API_KEY> <FILENAME>")
        sys.exit(1)
    api_key = sys.argv[1]
    file_name = sys.argv[2]
    os.environ["OPENAI_API_KEY"] = api_key

    # Load PDF file
    data = load_documents(file_name)

    # Chunk documents
    texts = chunk_documents(data)
    print(f'Now you have {len(texts)} documents')

    question = "what is tesla 2023 revenue"
    result = chromadb_retrieval_qa(texts, question)
    print(result["result"])
