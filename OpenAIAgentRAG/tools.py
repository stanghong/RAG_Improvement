import asyncio
import os

from pydantic import BaseModel

from agents import Agent, Runner, function_tool
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set OpenAI API key directly
os.environ["OPENAI_API_KEY"] = "sk-proj-your-api-key"

class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
def get_weather(city: str) -> Weather:
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()


def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)


@function_tool
def get_rag_answer(question: str) -> str:
    print("[debug] get_rag_answer called")
    # Load and chunk the documents
    data = load_documents('tesla10K.pdf')
    texts = chunk_documents(data)
    # Perform retrieval-augmented generation
    result = chromadb_retrieval_qa(texts, question)
    return result['result']


agent = Agent(
    name="Hello world", 
    instructions="You are a helpful agent.",
    tools=[get_weather, get_rag_answer],
)


async def main():
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")
        
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    result = await Runner.run(agent, input="What's tesla revenue in 2024?")
    result = await Runner.run(agent, input="who is ceo of tesla?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())
