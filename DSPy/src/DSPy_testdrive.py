import dspy
# %%
import pandas as pd

import dspy
import openai
openai.api_key = 'your API Key'

# %%
df=pd.read_csv('groundtruth_eval_dataset_500.csv')
# df.columns =['Unnamed: 0', 'question', 'context', 'answer', 'metadata']
df =df[['question', 'context', 'answer']]

# %%
def df2Dataset(df):
    dataset = []

    for question, context, answer in df.values:
        dataset.append(dspy.Example(question=question, context=context, answer=answer).with_inputs("context", "question"))
    return dataset

# %%
trainset=df2Dataset(df.iloc[:10])
devset=df2Dataset(df.iloc[11:21])
# %%
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

train_example = trainset[0]
print(f"Question: {train_example.question}")
print(f"Answer: {train_example.answer}")
# %%
devset
# %%
dev_example = devset[8]
print(f"Question: {dev_example.question}")
print(f"Answer: {dev_example.answer}")

# %%
print(f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys {train_example.labels().keys()}")
print(f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys {dev_example.labels().keys()}")
# %%
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
# %%
dev_example.question    
# %%
# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
pred = generate_answer(question=dev_example.question)

# Print the input and the prediction.
print(f"Question: {dev_example.question}")
print(f"Predicted Answer: {pred.answer}")
# %%
turbo.inspect_history(n=1)
# %%
# Define the predictor. Notice we're just changing the class. The signature BasicQA is unchanged.
generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)

# Call the predictor on the same input.
pred = generate_answer_with_chain_of_thought(question=dev_example.question)

# Print the input, the chain of thought, and the prediction.
print(f"Question: {dev_example.question}")
# print(f"Thought: {pred.rationale.split('.', 1)[1].strip()}")
print(f"Predicted Answer: {pred.answer}")
# %%
retrieve = dspy.Retrieve(k=3)
topK_passages = retrieve(dev_example.question).passages

print(f"Top {retrieve.k} passages for question: {dev_example.question} \n", '-' * 30, '\n')

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
# %%
retrieve("When was the first FIFA World Cup held?").passages[0]
# %%
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
# %%
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
# %%

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
# %%
from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)
