# Basic Agent Example with Weather and RAG Tools

This example demonstrates how to create and use an agent with custom tools for weather information and Retrieval-Augmented Generation (RAG).

## Setup

1. Set your OpenAI API key as an environment variable or directly in the code:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   ```

2. Install required dependencies:
   ```bash
   pip install langchain chromadb sentence-transformers unstructured
   ```

3. Place your Tesla 10K PDF file (named `tesla10K.pdf`) in the same directory as the script

## Running the Example

1. Run the script:
   ```bash
   python tools.py
   ```

2. The script will:
   - Create an agent with weather and RAG tools
   - Query the weather 
   - Query specific documents or knowledge base

3. Example outputs will be printed to the console, including:
   - Debug messages when tools are called
   - Weather information for Tokyo
   - Answers extracted from the Tesla 10K document
