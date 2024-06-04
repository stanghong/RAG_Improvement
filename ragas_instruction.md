
# Code Documentation

## Overview

This script involves a series of operations for analyzing data from a PDF report using Natural Language Processing (NLP) and Machine Learning techniques. The script is structured to process a 10K financial report of Tesla for the year 2023, evaluate the dataset, and generate Quality Control (QC) metrics for two pipelines.

### Libraries and Dependencies

- **Environment and API Access**: Utilizes `dotenv` for environment management and `openai` for accessing AI models.
- **Data Handling**: Employs `pandas` and `datasets` for data manipulation and storage.
- **PDF Processing**: Uses `pypdf` for reading PDF files.
- **Text Manipulation**: Involves `langchain` for advanced text splitting operations.
- **Machine Learning and NLP**: Implements `sentence_transformers` and `numpy` for text encoding and mathematical operations.
- **Utilities**: Custom utilities for text wrapping and pipeline operations are used.
- **Metric Evaluation**: Uses a custom module `ragas` for evaluating various metrics like relevancy and correctness.

## Usage

### Environment Setup

First, load the required environment variables:

```python
import os
from dotenv import load_dotenv

_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']
```

### Data Loading

Load the PDF file for the Tesla 2023 10K report:

```python
pdf_file = './data/tesla10K.pdf'
```

### Processing

Define the function to create a dataset for the RAG (Retrieval-Augmented Generation) process:

```python
def create_ragas_dataset(rag_pipeline, pdf_file, eval_dataset):
    # Implementation details
```

### Evaluation

Evaluate the RAG dataset using predefined metrics:

```python
def evaluate_ragas_service(ragas_dataset):
    # Implementation details
```

### Quality Control Metrics

Generate and save the QC metrics for different pipelines:

```python
ragas_dataset_pline1 = create_ragas_dataset(chromadb_retrieval_qa, pdf_file, eval_dataset)
evaluation_results_pline1 = evaluate_ragas_dataset(ragas_dataset_pline1)
df_pl1 = ragas_dataset_pline1.to_pandas()
df_pl1.to_excel('qc_metrics_pline1.xlsx')

ragas_dataset_pline2 = create_ragas_dataset(rag, pdf_file, eval_dataset)
evaluation_results_pline2 = evaluate_ragas_dataset(ragas_dataset_pline2)
df_pl2 = ragas_dataset_pline2.to_pandas()
df_pl2.to_excel('qc_cd_metrics_pline2.xlsx')
```

### Results Compilation

Compile results from both pipelines into a single Excel file for comparison:

```python
result = pd.concat([df_pl1, df_pl2], axis=1)
result.to_excel('merged.xlsx')
```

### Visualization

Transpose the results to present them in a readable format:

```python
df = pd.DataFrame({'ChromaDB': evaluation_results_pline1, 'ReRanker': evaluation.rvvisults_pline2})
df_transposed = df.transpose()
df_transposed
```

## Conclusion

This script demonstrates how to process, evaluate, and report on NLP-related data operations using advanced techniques in Python. It is tailored for analyzing financial documents, specifically Tesla's annual reports, and provides tools for comprehensive metric evaluation.
