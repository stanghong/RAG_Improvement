# %%
import dspy
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

load_dotenv()  # Load environment variables

turbo = dspy.OpenAI(model='gpt-3.5-turbo')

retriever_model = ChromadbRM(
    collection_name='tesla', 
    persist_directory="./teslasec",
    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    k=5
)

dspy.settings.configure(lm=turbo, rm=retriever_model)

# %%
from dspy.datasets import HotPotQA

# %%
import pandas as pd
df=pd.read_csv('../data/groundtruth_eval_dataset_300_titles.csv')
# df.columns =['Unnamed: 0', 'question', 'context', 'answer', 'metadata']
df =df[['question', 'context', 'answer']]

# %%
def df2Dataset(df):
    dataset = []

    for question, context, answer in df.values:
        dataset.append(dspy.Example(question=question, context=context, answer=answer).with_inputs('context', "question"))
    return dataset

# %%
trainset=df2Dataset(df.iloc[:20])
devset=df2Dataset(df.iloc[20:41])
# %%
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
# %%
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
# %%
from dsp.utils import deduplicate
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question, context=None):
        if context is None:
            context = []  # Ensure context is a list if not provided
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            retrieved_data = self.retrieve(query)
            passages = retrieved_data.passages if hasattr(retrieved_data, 'passages') else []

            # Ensure both context and passages are lists before concatenation
            if not isinstance(context, list):
                context = [context]
            if not isinstance(passages, list):
                passages = [passages]

            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)

# %%
# Ask any question you like to this simple RAG program.
my_question = "what's revenue of tesla 2022?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
pred = uncompiled_baleen(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
# %%
turbo.inspect_history(n=3)
# %%
#may need check openai_api_key
# %%

class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""
    
    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(desc="A rating between 1 and 5. Only output the rating and nothing else.")

metricLM = dspy.OpenAI(model='gpt-4', max_tokens=1000, model_type='chat')
# user defined metric
def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.answer  # Assuming `pred` is the prediction object and has an `answer` attribute
    question = gold.question

    print(f"Test Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")

    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"

    with dspy.context(lm=metricLM):
        context = gold.context
        detail = dspy.ChainOfThought(Assess)(context="N/A", assessed_question=detail, assessed_answer=predicted_answer)
        faithful = dspy.ChainOfThought(Assess)(context=context, assessed_question=faithful, assessed_answer=predicted_answer)
        overall = dspy.ChainOfThought(Assess)(context=context, assessed_question=overall, assessed_answer=predicted_answer)

    print(f"Faithful: {faithful.assessment_answer}")
    print(f"Detail: {detail.assessment_answer}")
    print(f"Overall: {overall.assessment_answer}")
    total = float(detail.assessment_answer) + float(faithful.assessment_answer)*2 + float(overall.assessment_answer)

    return total / 5.0

# %%
from dspy.teleprompt import BootstrapFewShot

# define optimizer using user defined metric
teleprompter = BootstrapFewShot(metric=llm_metric)
compiled_baleen = teleprompter.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset)
# %%
from dspy.evaluate.evaluate import Evaluate

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
# %%
uncompiled_baleen_retrieval_score = evaluate_on_hotpotqa(uncompiled_baleen, metric=llm_metric)
# %%
compiled_baleen_retrieval_score = evaluate_on_hotpotqa(compiled_baleen, metric=llm_metric)

print(f"## Retrieval Score for uncompiled Baleen: {uncompiled_baleen_retrieval_score}")
print(f"## Retrieval Score for compiled Baleen: {compiled_baleen_retrieval_score}")


# %%
