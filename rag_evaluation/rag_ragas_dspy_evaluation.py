# %%
# Libraries for environment and API access
import os
from dotenv import load_dotenv, find_dotenv
import openai

# Data handling and processing libraries
import pandas as pd
from datasets import Dataset

# PDF processing and text manipulation
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Machine Learning and NLP libraries
from sentence_transformers import CrossEncoder
import numpy as np
from tqdm import tqdm

# Utilities and custom functions
from helper_utils import word_wrap, create_ragas_dataset, evaluate_ragas_dataset
from RAG_pipeline1_chromadb import chromadb_retrieval_qa
from RAG_pipeline1_chromadb_finetune import chromadb_retrieval_qa_finetune
from RAG_pipeline2_crossencoder import rag, rank_doc
from dspy_eval_helper  import optimize_and_evaluate

# Metric evaluation
# from ragas import evaluate
# from ragas.metrics import (
#     answer_relevancy, faithfulness, context_recall, context_precision,
#     context_relevancy, answer_correctness, answer_similarity
# )

# Load environment variables
_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']

# %%
import datetime
now = datetime.datetime.now()
# Format datetime as a string (for example, '2024-06-08_12-00-00')
formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')

# # Load Tesla 2023 10K report
pdf_file='./data/tesla10K.pdf'

# %%
eval_dataset = Dataset.from_csv("./data/groundtruth_eval_dataset_174.csv")
eval_dataset=eval_dataset.select(range(15))

# %%
if 'ground_truths' in ragas_dataset_pline1.column_names:
    dataset = ragas_dataset_pline1.rename_column('ground_truths', 'ground_truth')

# %%
# eval pipeline 1 -----------------------------------------
ragas_dataset_pline1 = create_ragas_dataset(chromadb_retrieval_qa, pdf_file, eval_dataset )

# Make sure the new 'ground_truth' column is of the correct type (string)
# total row*7(metrics) runs to get RAGAS results
evaluation_results_pline1 = evaluate_ragas_dataset(ragas_dataset_pline1)

# Create filename with current datetime
df_pl1 = ragas_dataset_pline1.to_pandas()
df_pl1.to_excel(f'qc_metrics_pline1_{formatted_time}.xlsx')
df_pl1.to_pickle(f'qc_metrics_pline1_{formatted_time}.pkl')

# %%
# # eval pipeline 2 ChromaDB+GPT_finetune -----------------------------------------
ragas_dataset_pline2 = create_ragas_dataset(chromadb_retrieval_qa_finetune, pdf_file, eval_dataset)
evaluation_results_pline2 = evaluate_ragas_dataset(ragas_dataset_pline2)
# %%
# Create filename with current datetime
df_pl2 = ragas_dataset_pline2.to_pandas()
df_pl2.to_excel(f'qc_metrics_pline2_{formatted_time}.xlsx')
df_pl2.to_pickle(f'qc_metrics_pline2_{formatted_time}.pkl')
# %%
# eval pipeline 3: RAG -----------------------------------------
ragas_dataset_pline3 = create_ragas_dataset(rag, pdf_file, eval_dataset)
evaluation_results_pline3 = evaluate_ragas_dataset(ragas_dataset_pline3)
# %%
# Create filename with current datetime
df_pl3 = ragas_dataset_pline3.to_pandas()
df_pl3.to_excel(f'qc_metrics_pline3_{formatted_time}.xlsx')
df_pl3.to_pickle(f'qc_metrics_pline3_{formatted_time}.pkl')
# %%
# summarize results
result = pd.concat([df_pl1, df_pl2, df_pl3], axis=1)
result.to_excel(f'merged_{formatted_time}.xlsx')

# Create a DataFrame
df = pd.DataFrame({'ChromaDB_GPT ': evaluation_results_pline1, \
                   'ChromaDB_GPTft': evaluation_results_pline2,
                   'Reranker': evaluation_results_pline3})
df_transposed = df.transpose()
df_transposed
# %%
#eval part 2 -----------------------------------------------
# reproduce results without rerun above eval pipelines
# %%
formatted_time = '2024-06-27_14-56-03'  # Example, adjust it to your actual formatted_time
# Construct the file path with the formatted time
df_pl3 = Dataset.from_pandas(pd.read_pickle(f'qc_metrics_pline3_{formatted_time}.pkl'))
df_pl2 = Dataset.from_pandas(pd.read_pickle(f'qc_metrics_pline2_{formatted_time}.pkl'))
df_pl1 = Dataset.from_pandas(pd.read_pickle(f'qc_metrics_pline1_{formatted_time}.pkl'))
evaluation_results_pline3 = evaluate_ragas_dataset(df_pl3)
evaluation_results_pline2 = evaluate_ragas_dataset(df_pl2)
evaluation_results_pline1 = evaluate_ragas_dataset(df_pl1)

result=pd.read_excel('merged_2024-06-27_14-56-03.xlsx')
# Use your DataFrame
df = pd.DataFrame({'ChromaDB_GPT ': evaluation_results_pline1, \
                   'ChromaDB_GPTft': evaluation_results_pline2,
                   'Reranker': evaluation_results_pline3})
df_transposed = df.transpose()
df_transposed
# %%
# save evaluation_results_pline1
df_eval_rr=evaluation_results_pline3.to_pandas()
df_eval_ft=evaluation_results_pline2.to_pandas()
df_eval_gpt=evaluation_results_pline1.to_pandas()
# %%
df_eval_rr.to_excel('df_eval_rr.xlsx')
df_eval_ft.to_excel('df_eval_ft.xlsx')
df_eval_gpt.to_excel('df_eval_gpt.xlsx')
df_transposed.to_excel('df_eval_aggreted_scores.xlsx')

# %%
df_transposed=pd.read_excel('df_eval_aggreted_scores.xlsx')
new_row_names = ['ChromaDB_GPT', 'GPT_FineTune', 'Reranker']
df_transposed.index = new_row_names
df_transposed[['faithfulness', 'answer_relevancy',
        'answer_correctness',
       'answer_similarity']]

# %%
# use DSPy for model evaluation ---------------------------------
#DSPY Eval
metrics_fl_ggpt= 'qc_metrics_pline1_2024-06-27_14-56-03.xlsx'
result_gpt3, df_with_scores_gpt3=optimize_and_evaluate(metrics_fl_ggpt)
# %%
metrics_fl_ggptfinetune= 'qc_metrics_pline2_2024-06-27_14-56-03.xlsx'
result_gptft, df_with_scores_fptft=optimize_and_evaluate(metrics_fl_ggptfinetune)
# %%
metrics_fl_rr= 'qc_metrics_pline3_2024-06-27_14-56-03.xlsx'
result_reranker, df_with_scores_rr=optimize_and_evaluate(metrics_fl_rr)

# %%

# Create a DataFrame with the scalar values and an index
data = {
    'Score': [result_gpt3, result_reranker, result_gptft]
}
index = ['gpt3', 'reranker',  'fine tune']
df = pd.DataFrame(data, index=index)

# Print the DataFrame
print("DSPy Scores:")
print(df)
# %%
#save DSPy eval dataframe and scores table
df_with_scores_gpt3[['pred_answer', 'answer','total_score']].to_csv('gpt3.csv')
df_with_scores_fptft[['pred_answer', 'answer','total_score']].to_csv('ftpft.csv')
df_with_scores_rr[['pred_answer', 'answer','total_score']].to_csv('rr.csv')
# %%
