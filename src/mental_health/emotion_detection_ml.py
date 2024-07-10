from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
import os
import sys
from typing import List, Tuple, Callable, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import Preprocessor

class Dataset:
    def __init__(self):
        self.preprocessor = Preprocessor()
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
                'tokenize_sentence'
            ]

    def read_dialogue_data(self, file_path: str) -> List[List[str]]:
        '''
        Read the dialogue data from the file path.
        :param file_path: The path of the file.
        :type file_path: str
        :return: A list of dialogues, where each dialogue is a list of sentences,
                and each sentence is a string.
        :rtype: list
        '''
        # define dialogues list
        dialogues = []
        # read data file
        with open(file_path, "r", encoding="utf8") as file_data:
            dialogues = [
                [
                    sentence.replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(
                        ";", " ; ").replace(":", " : ").replace(",", " , ").strip()
                    for sentence in line.split("__eou__") if sentence.strip()
                ]
                for line in file_data
            ]
        return dialogues
    
    def read_dataset_dailyDialog(self, data_path: str, label_path: str) -> Tuple[List[List[List[str]]], List[List[int]]]:
        '''
        Take the data path and the label path and read them.
        It then splits the conversations and extracts each conversation, sentence, and words of each sentence.
        It reads the labels of each sentence in the conversation

        :param data_path: The path of the conversations.
        :type data_path: str
        :param label_path: The path of the labels for the conversations.
        :type label_path: str
        :return: A tuple containing inputs and targets.
                inputs: List of conversations, where each conversation is a list of sentences,
                        and each sentence is a list of words.
                targets: List of labels for each conversation.
        :rtype: tuple
        '''
        # define targets list
        targets = []
        # read labels file
        with open(label_path, "r", encoding="utf8") as file_data:
            targets = [[int(label) for label in line.strip(
                "\n").strip(" ").split(" ")] for line in file_data]

        # read data file
        dialogues = self.read_dialogue_data(data_path)
        # define inputs list
        inputs = [
            [
                self.preprocessor.clean(sentence, steps=self.steps) for sentence in dialogue
            ]
            for dialogue in dialogues
        ]
        return (inputs, targets)


class MentalHealthEmotionClassifier:
    def __init__(self,X_train,y_train,model, vectorizer='count'):
        self.model = model
        self.vectorizer_name = vectorizer
        self.pipeline = None
        self.X_train = [' '.join(sentence) for dialog in X_train for sentence in dialog]
        self.y_train = [label for dialog_labels in y_train for label in dialog_labels]
        self.emotion_map = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise'
        }
        self.preprocessor = Preprocessor()
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
                'tokenize_sentence'
            ]

    def _create_vectorizer(self):
        if self.vectorizer_name == 'count':
            self.vectorizer = CountVectorizer()
        elif self.vectorizer_name == 'tfidf':
            self.vectorizer = TfidfVectorizer()
    def save_model(self,model, save_path):
        """
        Save a trained model to a file using pickle.

        Parameters:
        - model: Trained model object
        - save_path: File path to save the model
        """
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    def _compute_class_weights(self):
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)

        # Convert class weights to a dictionary
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        return class_weight_dict

    def train(self, params, save_path):
        # Create vectorizer
        self._create_vectorizer()

        # Compute class weights
        class_weight_dict = self._compute_class_weights()

        # Set class weight in the model
        if hasattr(self.model, 'class_weight'):
            self.model.set_params(class_weight=class_weight_dict)

        # Build pipeline
        self.pipeline = Pipeline(steps=[('vectorizer', self.vectorizer), ('classifier', self.model)])

        # Set the parameters
        self.pipeline.set_params(**params)

        # Fit the model
        self.pipeline.fit(self.X_train, self.y_train)

        # Save the model
        self.save_model(self.pipeline, save_path)
        self.model = self.pipeline

    def predict(self, X_test, y_test):
        # Flatten X_test into a list of strings
        X_test = [' '.join(sentence) for dialog in X_test for sentence in dialog]
        y_test = [label for dialog_labels in y_test for label in dialog_labels]
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred)
        return report

    def tune_parameters(self, param_grid, save_path):
        # Define GridSearchCV
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

        # Perform Grid Search
        grid_search.fit(self.X_train, self.y_train)

        # Print best parameters
        print("Best parameters found:")
        print(grid_search.best_params_)

        # Save the best model
        best_model = grid_search.best_estimator_
        self.model = best_model
        save_model(best_model, save_path)

    
    def get_emotion_probabilities(self, preprocessed_text):
        preprocessed_text_str = " ".join(preprocessed_text)
        vectorized_sentence = self.model.named_steps['vectorizer'].transform([preprocessed_text_str])
        predicted_emotion = self.model.named_steps['classifier'].predict(vectorized_sentence)[0]
        class_probabilities = self.model.named_steps['classifier'].predict_proba(vectorized_sentence)[0]
        emotion_probabilities = {self.emotion_map[i]: prob for i, prob in enumerate(class_probabilities)}
        return emotion_probabilities, self.emotion_map[predicted_emotion]
    def load_model(self,load_path):
        """
        Load a trained model from a file using pickle.

        Parameters:
        - load_path: File path of the saved model

        Returns:
        - model: Trained model object
        """
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        return model

if __name__ == "__main__":


    dataset_path = "D:/College/Fourth Year/GP/dailydialog"

    train_data_path = f"{dataset_path}/train/dialogues_train.txt"
    train_label_path = f"{dataset_path}/train/dialogues_emotion_train.txt"
    test_data_path = f"{dataset_path}/test/dialogues_test.txt"
    test_label_path = f"{dataset_path}/test/dialogues_emotion_test.txt"

    # Read the dataset
    dataset = Dataset()
    X_train, y_train = dataset.read_dataset_dailyDialog(train_data_path, train_label_path)
    X_test, y_test = dataset.read_dataset_dailyDialog(test_data_path, test_label_path)

    models = [
        (LogisticRegression(), 'count', {'classifier__C': 10}, "lr"),
        (DecisionTreeClassifier(), 'count', {'classifier__max_depth': 20}, "dt"),
        (XGBClassifier(), 'tfidf', {'classifier__n_estimators': 150}, "xgboost"),
        (RandomForestClassifier(), 'tfidf', {'classifier__n_estimators': 100}, "rf")
    ]
    isTrain = True
    for model, vectorizer, params, model_type in models:
        classifier = MentalHealthEmotionClassifier(X_train, y_train, model=model, vectorizer=vectorizer)
        save_path = f"D:/College/Fourth Year/GP/Soul-AI/models/{model_type}_ml_main.pkl"

        if isTrain:
            classifier.train(params, save_path)
        else:
            classifier.load_model(save_path)
        
        # Predict on test set
        report = classifier.predict(X_test, y_test)
        print(f"Classification report for {model_type}:\n", report)
        
        # Example of predicting a new sentence
        new_sentence = "Okay, I'll give it a try. But I'm still scared and doubtful."
        preprocessed_text = classifier.preprocessor.clean(new_sentence, steps=classifier.steps)
        probabilities, predicted_emotion = classifier.get_emotion_probabilities(preprocessed_text)
        print(f"Sentence =  {new_sentence}")
        print(f"Emotion probabilities for {model_type}:", probabilities)
        print(f"Predicted emotion for {model_type}:", predicted_emotion)

        new_sentence = "I feel happy today!"
        preprocessed_text = classifier.preprocessor.clean(new_sentence, steps=classifier.steps)
        probabilities, predicted_emotion = classifier.get_emotion_probabilities(preprocessed_text)
        print(f"Sentence =  {new_sentence}")
        print(f"Emotion probabilities for {model_type}:", probabilities)
        print(f"Predicted emotion for {model_type}:", predicted_emotion)

        new_sentence = "After a difficult day, I'm feeling a bit overwhelmed and sad."
        preprocessed_text = classifier.preprocessor.clean(new_sentence, steps=classifier.steps)
        probabilities, predicted_emotion = classifier.get_emotion_probabilities(preprocessed_text)
        print(f"Sentence =  {new_sentence}")
        print(f"Emotion probabilities for {model_type}:", probabilities)
        print(f"Predicted emotion for {model_type}:", predicted_emotion)

        new_sentence = " I can't decide whether to be angry or just disappointed."
        preprocessed_text = classifier.preprocessor.clean(new_sentence, steps=classifier.steps)
        probabilities, predicted_emotion = classifier.get_emotion_probabilities(preprocessed_text)
        print(f"Sentence =  {new_sentence}")
        print(f"Emotion probabilities for {model_type}:", probabilities)
        print(f"Predicted emotion for {model_type}:", predicted_emotion)