from typing import List, Tuple, Dict
import random
import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorizers.count_vectorizer import CountVectorizer
from preprocess import Preprocessor
from helper import Helper

def cosine_similarity(vec_a, vec_b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vec_a (numpy array): The first vector.
    vec_b (numpy array): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    dot_product = np.dot(vec_a, vec_b)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0
    else:
        return dot_product / (norm_a * norm_b)

class Suicidal:
    def __init__(self,
                 dataset_path='../../data/raw/suicidal_detection',
                 model_path='../../models/suicidal',
                 isTrain=False,
                 split_seed=10,
                 seed=2,
                 model_name='model_centroids',
                 ):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.isTrain = isTrain
        self.split_seed = split_seed
        self.seed = seed
        self.limit = 10019

        self.model_name = model_name
        self.data_path = f"{self.dataset_path}/Suicide_Detection_Reddit.csv"
        self.train_preprocess = f"{self.dataset_path}/train/dialogues_train_preprocess_ml.pkl"
        self.test_preprocess = f"{self.dataset_path}/test/dialogues_test_preprocess_ml.pkl"
        self.model_save_path = f"{self.model_path}/{self.model_name}.pkl"

        self.forward_label_mapping = {'suicide': 1, 'non-suicide': 0}
        self.reverse_label_mapping = {0: 'non-suicide', 1: 'suicide'}

        self.helper = Helper()
        self.preprocessor = Preprocessor()

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.steps = [
            'translate_emojis_to_text',
            'lower_sentence',
            'remove_nonascii_diacritic',
            'remove_emails',
            'clean_html',
            'remove_url',
            'replace_repeated_chars',
            'expand_sentence',
            'remove_possessives',
            'remove_extra_space',
        ]

    def read_dataset_suicidal_detection(self, data_path: str, split_seed: int) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
        data = pd.read_csv(data_path)
        dialogues = data['text'].tolist()
        labels = data['class'].apply(lambda x: self.forward_label_mapping[x]).tolist()

        dialogues_train, dialogues_test, labels_train, labels_test = train_test_split(
            dialogues, labels, test_size=0.2, random_state=split_seed)

        return (dialogues_train, labels_train), (dialogues_test, labels_test)

    def clean_data(self, data: List[str]) -> List[List[str]]:
        cleaned_data = []
        for line in data:
            cleaned_line = self.preprocessor.clean(line, steps=self.steps)
            cleaned_data.append(cleaned_line)
        return cleaned_data

    def train_model(self, dialogues: List[str], labels: List[int]):
        dialogues = dialogues[:self.limit]
        labels = labels[:self.limit]
        dialogues = [temp[0] for temp in dialogues]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(dialogues)

        # Compute centroids for each class
        centroid_0 = np.mean([X[i] for i in range(len(X)) if labels[i] == 0], axis=0)
        centroid_1 = np.mean([X[i] for i in range(len(X)) if labels[i] == 1], axis=0)

        self.helper.dump_tuple(self.model_save_path, (vectorizer, centroid_0, centroid_1))

    def evaluate_model(self, dialogues: List[str], labels: List[int]):
        dialogues = dialogues[:self.limit]
        labels = labels[:self.limit]
        dialogues = [temp[0] for temp in dialogues]
        vectorizer, centroid_0, centroid_1 = self.helper.load_tuple(self.model_save_path)
        X = vectorizer.transform(dialogues)
        predictions = []

        for x in X:
            similarity_0 = cosine_similarity(x, centroid_0)
            similarity_1 = cosine_similarity(x, centroid_1)
            predictions.append(0 if similarity_0 > similarity_1 else 1)

        report = classification_report(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        return accuracy, report

    def dev_mode(self):
        print(f'starting dev mode with train is set to {self.isTrain}')
        # read the data set
        (dialogues_train, labels_train),  (dialogues_test, labels_test) = self.read_dataset_suicidal_detection(self.data_path, self.split_seed)
        if self.isTrain:
            if not os.path.exists(self.train_preprocess):
                print("Preprocessing the training data")
                dialogues_train_proccessed = self.clean_data(dialogues_train)
                self.helper.dump_tuple(self.train_preprocess,(dialogues_train_proccessed, labels_train))
            else:
                print("The preprocessed train data already exists")
                dialogues_train_proccessed, labels_train = self.helper.load_tuple(self.train_preprocess)

        if not os.path.exists(self.test_preprocess):
            print("Preprocessing the testing data")
            dialogues_test_proccessed = self.clean_data(dialogues_test)
            self.helper.dump_tuple(self.test_preprocess, (dialogues_test_proccessed, labels_test))
        else:
            print("The preprocessed test data already exists")
            dialogues_test_proccessed, labels_test = self.helper.load_tuple(self.test_preprocess)
        # train the model
        if self.isTrain:
            self.train_model(dialogues_train_proccessed, labels_train)

        accuracy, report = self.evaluate_model(dialogues_test_proccessed, labels_test)
        print(f"Test Accuracy: {accuracy}")
        print(report)

    def suicidal_detection(self, sentence: str) -> Tuple[str, Dict[str, float]]:
        vectorizer, centroid_0, centroid_1 = self.helper.load_tuple(self.model_save_path)
        cleaned_sentence = self.preprocessor.clean(sentence, steps=self.steps)
        cleaned_sentence = cleaned_sentence[0]
        X = vectorizer.transform([cleaned_sentence])

        # Compute cosine similarity with each centroid
        similarity_0 = cosine_similarity(X, centroid_0)
        similarity_1 = cosine_similarity(X, centroid_1)

        # Determine the class based on the highest similarity
        if similarity_0 > similarity_1:
            predicted_class_name = 'non-suicide'
            class_probabilities = {
                'non-suicide': similarity_0,
                'suicide': similarity_1
            }
        else:
            predicted_class_name = 'suicide'
            class_probabilities = {
                'non-suicide': similarity_0,
                'suicide': similarity_1
            }

        return predicted_class_name, class_probabilities

    def prod_mode(self):
        while True:
            exit_key = input("Enter q to exit else to continue: ")
            if exit_key == 'q':
                break
            sentence = input("Enter the sentence: ")
            predicted_class_name, class_probabilities = self.suicidal_detection(sentence)
            print(f"Sentence: {sentence}")
            print(f"Predicted class: {predicted_class_name}")
            print("Class probabilities:")
            for class_name, probability in class_probabilities.items():
                print(f"{class_name}: {probability}")
            print('----------------------------------------------')


if __name__ == '__main__':
    while True:
        print('''
            Choose the mode you want to operate...
            1. Train Mode
            2. Test Mode
            3. Production Mode
            else to exit
            ''')
        mode = input("Enter your choice: ")
        if mode == '1':
            suicidal_model = Suicidal(isTrain=True)
            suicidal_model.dev_mode()
        elif mode == '2':
            suicidal_model = Suicidal()
            suicidal_model.dev_mode()
        elif mode == '3':
            suicidal_model = Suicidal()
            suicidal_model.prod_mode()
        else:
            print('Exit...')
            exit()