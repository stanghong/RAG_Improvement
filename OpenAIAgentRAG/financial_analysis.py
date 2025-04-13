import asyncio
import os

from pydantic import BaseModel

from agents import Agent, Runner, function_tool
from rag import chromadb_retrieval_qa, load_documents, chunk_documents

# Set OpenAI API key directly 
os.environ["OPENAI_API_KEY"] = "sk-proj-youropenaiapikey"


@function_tool
def get_rag_answer(question: str) -> str:
    print("[debug] get_rag_answer called")
    data = load_documents('tesla10K.pdf')
    texts = chunk_documents(data)
    result = chromadb_retrieval_qa(texts, question)
    return result['result']


@function_tool
def summarize_document() -> str:
    print("[debug] summarize_document called")
    data = load_documents('tesla10K.pdf')
    texts = chunk_documents(data)
    # Simulate summarization
    return "Summary of the document: Key financial metrics and projections."


@function_tool
def compare_with_historical() -> str:
    print("[debug] compare_with_historical called")
    # Simulate comparison
    return "Comparison with historical data: Revenue growth is consistent with past trends."


agent = Agent(
    name="Comprehensive Financial Analysis Agent",
    instructions="You are a financial analysis agent. Use the RAG tool and other tools to provide a comprehensive analysis.",
    tools=[get_rag_answer, summarize_document, compare_with_historical],
)


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Demonstrate the use of the RAG tool to answer a specific question
    result = await Runner.run(agent, input="What's Tesla's revenue in 2023?")
    print("RAG Tool Output:", result.final_output)
    
    # Demonstrate the use of the summarize_document tool
    result = await Runner.run(agent, input="Summarize the document. add few more details")
    print("Summarize Document Tool Output:", result.final_output)
    
    # Demonstrate the use of the compare_with_historical tool
    result = await Runner.run(agent, input="Compare with historical data. add few more details")
    print("Compare with Historical Data Tool Output:", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
