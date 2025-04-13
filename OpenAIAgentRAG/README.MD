# Financial Analysis Agent with RAG and Analysis Tools

This example demonstrates how to create and use an agent with custom tools for financial analysis using Retrieval-Augmented Generation (RAG) and other analytical capabilities.

why do we use agent +tools instead of prompt engineering
|----------------------------|----------------------------------------|
| Prompt Engineering         | Agent + Tools                          |
|----------------------------|----------------------------------------|
| LLM-only logic             | Combines LLM + custom code             |
| Limited to model knowledge | Access to real-time data, files, APIs  |
| No structured memory or logic | You can build workflows and logic chains |
| Manual prompt chaining     | Auto-decision based on inputs          |
| Not scalable               | Super modular and extendable           |
|----------------------------|----------------------------------------|


## Setup

1. Set your OpenAI API key as an environment variable or directly in the code:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   ```

2. Install required dependencies:
   ```bash
   pip install langchain langchain-community chromadb sentence-transformers unstructured
   ```

3. Place your Tesla 10K PDF file (named `tesla10K.pdf`) in the same directory as the script

## Components

1. RAG Implementation (`rag.py`):
   - Loads and chunks PDF documents
   - Uses SentenceTransformer embeddings and ChromaDB for vector storage
   - Implements question-answering using GPT-3.5-turbo

2. Financial Analysis Agent (`financial_analysis.py`):
   - Comprehensive financial analysis tools
   - RAG-based question answering
   - Document summarization
   - Historical data comparison

## Running the Example

1. Run the financial analysis script:
   ```bash
   python financial_analysis.py
   ```

2. The script demonstrates:
   - RAG-based querying of Tesla's 2023 revenue
   - Document summarization with key financial metrics
   - Historical data comparison analysis

3. Example outputs include:
   - RAG Tool responses to specific financial queries
   - Document summary with key financial metrics
   - Historical trend analysis and comparisons
   - Debug messages for tool execution
