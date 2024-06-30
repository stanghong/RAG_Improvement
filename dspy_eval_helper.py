# %%
import os
import pandas as pd
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# Load environment settings
def load_environment():
    load_dotenv('.env')
    dspy.settings.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250))

# Load and prepare data from an Excel file
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(columns={'answer': 'pred_answer', 'contexts': 'context', 'ground_truths': 'answer'})
    df = df[['question', 'pred_answer', 'context', 'answer']]
    return df

def convert_to_dataset(df):
    dataset = []
    for index, row in df.iterrows():
        dataset.append(dspy.Example(question=row['question'], pred_answer=row['pred_answer'], context=row['context'], answer=row['answer']).with_inputs("question"))
    return dataset

# Model Definitions
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

class Assess(dspy.Signature):
    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(desc="A rating between 1 and 5. Only output the rating and nothing else.")

def llm_metric(gold, pred):
    predicted_answer = gold.pred_answer # pass RAG answer directly, later will connect to rag pipeline
    question = gold.question
    metricLM = dspy.OpenAI(model='gpt-4o', max_tokens=1000, model_type='chat')
    with dspy.context(lm=metricLM):
        context = gold.context
        detail_score = dspy.ChainOfThought(Assess)(context="N/A", assessed_question="Is the assessed answer detailed?", assessed_answer=predicted_answer)
        faithful_score = dspy.ChainOfThought(Assess)(context=context, assessed_question="Is the assessed text grounded in the context? Say no if it includes significant facts not in the context.", assessed_answer=predicted_answer)
        overall_score = dspy.ChainOfThought(Assess)(context=context, assessed_question=f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`", assessed_answer=predicted_answer)
        total_score = (float(detail_score.assessment_answer) + float(faithful_score.assessment_answer) * 2 + float(overall_score.assessment_answer)) / 4
        return total_score

# Function to optimize and evaluate, returning average total score and DataFrame with individual scores
def optimize_and_evaluate(file_path):
    df = load_and_prepare_data(file_path)
    dataset = convert_to_dataset(df)
    
    total_scores = []
    for index, example in enumerate(dataset):
        total_score = llm_metric(example, example)
        df.at[index, 'total_score'] = total_score
        total_scores.append(total_score)
    
    average_total_score = sum(total_scores) / len(total_scores) if total_scores else 0
    return average_total_score, df  # Returning the average total score and the DataFrame with scores

if __name__ == '__main__':
    load_environment()
    # file_path = 'qc_metrics_pline3_2024-06-27_14-56-03.xlsx' #rerranker
    # file_path = 'qc_metrics_pline2_2024-06-27_14-56-03.xlsx' #finetune
    file_path = 'qc_metrics_pline1_2024-06-27_14-56-03.xlsx'  #gpt3.5
    average_total_score, df_with_scores = optimize_and_evaluate(file_path)
    print("Average Total Score:", average_total_score)
    print("DataFrame with Scores:\n", df_with_scores)
