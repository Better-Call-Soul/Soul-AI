import pandas as pd
from rouge_score import rouge_scorer

# Summarization Evaluator class
# This class is used to evaluate the performance of a summarization model
class SummarizationEvaluator:
    # initialize the model and dataset path
    def __init__(self, model,dataset_path="../data/raw/summarization/bbc-news-summary/bbc-news-summary.csv"):
        self.dataset_path = dataset_path
        self.model = model
        self.reference_summaries = []
        self.generated_summaries = []

    # read the dataset
    # Read the dataset from a CSV file and populate the reference summaries.
    def read_dataset(self):
        df = pd.read_csv(self.dataset_path)
        self.documents = df['Articles'].tolist()[:150]
        self.reference_summaries = df['Summaries'].tolist()[:150]
        
        # print(f"documents from the dataset. {(self.documents)} ")
        # print(f"reference summaries from the dataset. {(self.reference_summaries)} ")
        
    # summarize the data
    def summarize(self):
        self.generated_summaries = [self.model.summary(document) for document in self.documents]
        # print(f"summaries using the model. {(self.generated_summaries)} ")

    # calculate the average ROUGE scores
    def calculate_avg_rouge_scores(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        # loop for each reference and generated summary
        for ref, gen in zip(self.reference_summaries, self.generated_summaries):
            score = scorer.score(ref, gen)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        # calculate the average scores
        avg_scores = {key: sum(value)/len(value) for key, value in scores.items()}
        return avg_scores
    
    # evaluate the model
    def evaluate(self):
        self.read_dataset()
        self.summarize()
        avg_scores = self.calculate_avg_rouge_scores()
        return avg_scores

