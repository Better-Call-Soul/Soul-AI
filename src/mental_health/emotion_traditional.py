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
    vec_a = np.array(vec_a).flatten()
    vec_b = np.array(vec_b).flatten()

    dot_product = np.dot(vec_a, vec_b)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Ensure this is a scalar float
    else:
        return float(dot_product / (norm_a * norm_b))  # Ensure this is a scalar float


class Emotion:

    def __init__(self,
                 dataset_path='../../data/raw/detection/DailyDialog/dailydialog',
                 model_path='../../models',
                 isTrain=False,
                 seed=2,
                 model_name='model_centroids',
                 num_classes = 7
                 ):

        self.dataset_path = dataset_path
        self.model_path = model_path
        self.isTrain = isTrain
        self.seed = seed
        self.limit = 100

        self.model_name = model_name
        self.train_data_path = f"{self.dataset_path}/train/dialogues_train.txt"
        self.train_label_path = f"{self.dataset_path}/train/dialogues_emotion_train.txt"
        self.dev_data_path = f"{self.dataset_path}/validation/dialogues_validation.txt"
        self.dev_label_path = f"{self.dataset_path}/validation/dialogues_emotion_validation.txt"
        self.test_data_path = f"{self.dataset_path}/test/dialogues_test.txt"
        self.test_label_path = f"{self.dataset_path}/test/dialogues_emotion_test.txt"
        self.train_preprocess = f"{self.dataset_path}/train/dialogues_train_preprocess_ml.pkl"
        self.dev_preprocess = f"{self.dataset_path}/validation/dialogues_validation_preprocess_ml.pkl"
        self.test_preprocess = f"{self.dataset_path}/test/dialogues_test_preprocess_ml.pkl"
        self.model_save_path = f"{self.model_path}/dailyDialog/{self.model_name}.pt"

        self.num_classes = num_classes


        self.helper = Helper()
        self.preprocessor = Preprocessor()

        # fix random seeds
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
        # Define mapping of classes to emotions
        self.emotion_map = {
                0: 'neutral',
                1: 'anger',
                2: 'disgust',
                3: 'fear',
                4: 'happiness',
                5: 'sadness',
                6: 'surprise'
            }

    def read_dialogue_data(self, file_path: str) -> List[str]:
        '''
        Read the dialogue data from the file path and return all sentences in a single list.
        :param file_path: The path of the file.
        :type file_path: str
        :return: A list of sentences, where each sentence is a string.
        :rtype: list
        '''
        sentences = []
        with open(file_path, "r", encoding="utf8") as file_data:
            for line in file_data:
                sentences.extend([
                    sentence.replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(
                        ";", " ; ").replace(":", " : ").replace(",", " , ").strip()
                    for sentence in line.split("__eou__") if sentence.strip()
                ])
        return sentences
    
    
    def read_dataset_dailyDialog(self, data_path: str, label_path: str) -> Tuple[List[List[str]], List[int]]:
        '''
        Take the data path and the label path and read them.
        It then splits the conversations and extracts sentences, converting each sentence into a list of words.
        It reads the labels of each sentence in the conversation.

        :param data_path: The path of the conversations.
        :type data_path: str
        :param label_path: The path of the labels for the conversations.
        :type label_path: str
        :return: A tuple containing inputs and targets.
                inputs: List of sentences, where each sentence is a list of words.
                targets: List of labels for each sentence.
        :rtype: tuple
        '''
        # Read labels file and flatten the list
        targets = []
        with open(label_path, "r", encoding="utf8") as file_data:
            for line in file_data:
                targets.extend([int(label) for label in line.strip("\n").strip(" ").split(" ")])

        # Read data file and flatten the dialogues into sentences
        dialogues = self.read_dialogue_data(data_path)
        inputs = []
        for dialogue in dialogues:
            for sentence in dialogue:
                cleaned_sentence = self.preprocessor.clean(sentence, steps=self.steps)[0]
                inputs.append(cleaned_sentence)

        return (inputs, targets)

    def train_model(self, dialogues: List[str], labels: List[int]):
        dialogues = dialogues[:self.limit]
        labels = labels[:self.limit]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(dialogues)
        
    
        # Ensure X is a NumPy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
    
        # Compute centroids for each class with for loop on num of class
        centroids = []
        for i in range(self.num_classes):
            class_vectors = [X[j] for j in range(len(X)) if labels[j] == i]
            if class_vectors:
                centroid = np.mean(class_vectors, axis=0)
            else:
                centroid = np.zeros(X.shape[1])  # Use zeros if no vectors are present
            centroids.append(centroid)
        
        self.helper.dump_tuple(self.model_save_path, (vectorizer, centroids))

    def evaluate_model(self, dialogues: List[str], labels: List[int]):
        dialogues = dialogues[:self.limit]
        labels = labels[:self.limit]
        vectorizer, centroids = self.helper.load_tuple(self.model_save_path)
        X = vectorizer.transform(dialogues)
        predictions = []

        for x in X:
            similarities = [cosine_similarity(x, centroid) for centroid in centroids]
            predicted_class = np.argmax(similarities)
            predictions.append(predicted_class)

        report = classification_report(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        return accuracy, report

    def dev_mode(self):
        print(f'starting dev mode with train is set to {self.isTrain}')

        if self.isTrain:
            if not os.path.exists(self.train_preprocess):
                print("Preprocessing the training data")
                inputs, targets = self.read_dataset_dailyDialog(
                    self.train_data_path, self.train_label_path)
                self.helper.dump_tuple(self.train_preprocess, (inputs, targets))
            else:
                print("The preprocessed train data already exists")


        if not os.path.exists(self.test_preprocess):
            print("Preprocessing the testing data")
            inputs, targets = self.read_dataset_dailyDialog(self.test_data_path, self.test_label_path)
            self.helper.dump_tuple(self.test_preprocess, (inputs, targets))
        else:
            print("The preprocessed test data already exists")

        test_data, test_label = self.helper.load_tuple(self.test_preprocess)


        # train the model
        if self.isTrain:
            train_data, train_label = self.helper.load_tuple(self.train_preprocess)
            self.train_model(train_data, train_label)

        accuracy, report = self.evaluate_model(test_data, test_label)
        print(f"Test Accuracy: {accuracy}")
        print(report)

    def emotion_detection(self, sentence: str) -> Tuple[str, Dict[str, float]]:
        vectorizer, centroids = self.helper.load_tuple(self.model_save_path)
        cleaned_sentence = self.preprocessor.clean(sentence, steps=self.steps)
        cleaned_sentence = cleaned_sentence[0]
        X = vectorizer.transform([cleaned_sentence])
    
        # Compute cosine similarity with each centroid
        similarities = [cosine_similarity(X, centroid) for centroid in centroids]
    
        # Debug: Print the similarities to inspect their values
        print("Similarities:", similarities)
    
        # Ensure similarities are scalar values
        if not all(isinstance(sim, (int, float)) for sim in similarities):
            raise ValueError("Cosine similarities must be scalar values.")
    
        # Determine the class based on the highest similarity
        predicted_class = np.argmax(similarities)
        class_probabilities = {self.emotion_map[i]: round((similarity * 100), 3) for i, similarity in enumerate(similarities)}
    
        return self.emotion_map[predicted_class], class_probabilities

    def prod_mode(self):
        while True:
            exit_key = input("Enter q to exit else to continue: ")
            if exit_key == 'q':
                break
            sentence = input("Enter the sentence: ")
            predicted_class_name, class_probabilities = self.emotion_detection(sentence)
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
            emotion_model = Emotion(isTrain=True)
            emotion_model.dev_mode()
        elif mode == '2':
            emotion_model = Emotion()
            emotion_model.dev_mode()
        elif mode == '3':
            emotion_model = Emotion()
            emotion_model.prod_mode()
        else:
            print('Exit...')
            exit()
