from typing import List, Tuple, Callable, Dict
import argparse
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import torch.autograd
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
from torchnlp.word_to_vector import GloVe
import contractions
import unicodedata
from bs4 import BeautifulSoup
import emoji
import re
from spellchecker import SpellChecker
import pickle
import matplotlib.pyplot as plt
import threading
from joblib import Parallel, delayed
import sys
import os

print("Finished importing libraries")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess import Preprocessor
from helper import Helper
from glove import Glove

from model import *
from dataset import *
from metric import *

print("Finished importing custom modules")


class Emotion:

    def __init__(self,
                dataset_path='../../data/raw/detection/DailyDialog/dailydialog',
                model_path='../../models',
                isTrain=False,
                seed=2,
                model_name='model_100E',
                batch_size=16,
                embedding_size=300,
                lstm_hidden_size=500,
                hidden_layer_size=512,
                epochs=1,
                num_classes=7,
                learning_rate=0.001,
                word2vec=Glove(),
                ):
        
        # The folder of the data set
        self.dataset_path = dataset_path
        # The folder for the models
        self.model_path = model_path
        self.isTrain = isTrain
        self.seed = seed
        self.model_name = model_name

        self.train_data_path = f"{self.dataset_path}/train/dialogues_train.txt"
        self.train_label_path = f"{self.dataset_path}/train/dialogues_emotion_train.txt"
        self.dev_data_path = f"{self.dataset_path}/validation/dialogues_validation.txt"
        self.dev_label_path = f"{self.dataset_path}/validation/dialogues_emotion_validation.txt"
        self.test_data_path = f"{self.dataset_path}/test/dialogues_test.txt"
        self.test_label_path = f"{self.dataset_path}/test/dialogues_emotion_test.txt"
        self.train_preprocess = f"{self.dataset_path}/train/dialogues_train_preprocess.pkl"
        self.dev_preprocess = f"{self.dataset_path}/validation/dialogues_validation_preprocess.pkl"
        self.test_preprocess = f"{self.dataset_path}/test/dialogues_test_preprocess.pkl"
        self.model_save_path = f"{self.model_path}/dailyDialog/{self.model_name}.pt"

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.helper = Helper()
        self.preprocessor = Preprocessor()
        self.word2vec = word2vec

        # fix random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)


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

    
    def read_dataset_dailyDialog_old(self, data_path: str, label_path: str) -> Tuple[List[List[List[str]]], List[List[int]]]:
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
        # define inputs list
        inputs = []
        # define targets list
        targets = []
        # read data file
        with open(data_path, "r", encoding="utf8") as file_data:
            inputs = [
                [sentence.split(" ") for sentence in line.strip(
                    "\n").strip(" __eou__").split(" __eou__")]
                for line in file_data
            ]
            # for loop version
            # for line in file_data:
            #   _sentences = line.strip("\n").strip(" __eou__").split(" __eou__")
            #   sentences = []
            #   for sentence in _sentences:
            #     words = sentence.split(" ")
            #     sentences.append(words)
            #   inputs.append(sentences)
        # read labels file
        with open(label_path, "r", encoding="utf8") as file_data:
            targets = [[int(label) for label in line.strip(
                "\n").strip(" ").split(" ")] for line in file_data]
            # for loop version
            # for line in file_data:
            #     labels = [int(label) for label in line.strip("\n").strip(" ").split(" ")]
            #     targets.append(labels)

        return (inputs, targets)
    

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
            # for loop version
            # for line in file_data:
            #     labels = [int(label) for label in line.strip("\n").strip(" ").split(" ")]
            #     targets.append(labels)

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
    

    

    def read_dataset_dailyDialog_threads(self, data_path: str, label_path: str) -> Tuple[List[List[List[str]]], List[List[int]]]:
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
        # read labels file
        with open(label_path, "r", encoding="utf8") as file_data:
            targets = [[int(label) for label in line.strip(
                "\n").strip(" ").split(" ")] for line in file_data]

        # read data file
        dialogues = self.read_dialogue_data(data_path)

        # Define function to clean sentences
        def clean_sentences(dialogue):
            return [self.preprocessor.clean(sentence, steps=self.steps) for sentence in dialogue]

        # Process data (parallelizing sentence cleaning)
        inputs = Parallel(n_jobs=-1)(
            delayed(clean_sentences)(dialogue) for dialogue in dialogues
        )

        return (inputs, targets)
    

    def dataset_embedding(self, inputs: List[List[List[str]]], word2vec: Callable[[str], torch.Tensor]) -> List[List[List[str]]]:
        '''
        Convert the dataset to embeddings.
        :param inputs: The inputs of the dataset.
        :type inputs: list
        :return: A list containing the embeddings.
        :rtype: list
        '''
        # define the embeddings list
        embeddings = []
        # for loop version
        for dialogue in inputs:
            # define the dialogue embeddings list
            dialogue_embeddings = []
            for sentence in dialogue:
                # define the sentence embeddings list
                sentence_embeddings = []
                for word in sentence:
                    try:
                        # get the word embedding
                        word_embedding = word2vec(word)
                        # append the word embedding to the sentence embeddings
                        sentence_embeddings.append(word_embedding)
                    except KeyError:
                        # if the word is not in the vocabulary, append a random vector
                        sentence_embeddings.append(torch.rand(self.embedding_size))
                # append the sentence embeddings to the dialogue embeddings
                dialogue_embeddings.append(sentence_embeddings)
            # append the dialogue embeddings to the embeddings list
            embeddings.append(dialogue_embeddings)
        # return the embeddings
        return embeddings
    

    def train_model(self, model: nn.Module, train_data_loader: DataLoader, loss_criterion: nn.Module, optimizer: torch.optim.Optimizer ) -> Tuple[float, float, float, float, float, str]:
        '''
        Train the model using the training data.
        :param model: The model to train.
        :type model: nn.Module
        :param train_data_loader: The DataLoader for the training data.
        :type train_data_loader: DataLoader
        :param loss_criterion: The loss criterion.
        :type loss_criterion: nn.Module
        :param optimizer: The optimizer.
        :type optimizer: torch.optim.Optimizer
        :return: A tuple containing the loss, accuracy, precision, recall, F1 score, and the report.
        :rtype: tuple
        '''
        model.train()
        loss_values = []
        predicted_labels_list = []
        true_labels_list = []

        for input_data, target_labels, masks in train_data_loader:
            optimizer.zero_grad()
            model_output = model(input_data)

            # Flatten model_output and target_labels while considering masks
            model_output_flat = model_output[masks].view(-1, model_output.size(-1))
            target_labels_flat = target_labels[masks].view(-1)

            # Calculate loss
            loss = loss_criterion(model_output_flat, target_labels_flat)

            # Backpropagation
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            predicted_labels_list.append(torch.argmax(model_output_flat, dim=-1).cpu().numpy())
            true_labels_list.append(target_labels_flat.cpu().numpy())

        predicted_labels = np.concatenate(predicted_labels_list)
        true_labels = np.concatenate(true_labels_list)

        average_loss = np.mean(loss_values)
        evaluation_scores = calculate_evaluation_scores(true_labels, predicted_labels)

        return (average_loss, *evaluation_scores)
    

    def evaluate_model(self, model: nn.Module, eval_data_loader: DataLoader, loss_criterion: nn.Module) -> Tuple[float, float, float, float, float, str]:
        '''
        Evaluate the model using the evaluation data.
        :param model: The model to evaluate.
        :type model: nn.Module
        :param eval_data_loader: The DataLoader for the evaluation data.
        :type eval_data_loader: DataLoader
        :param loss_criterion: The loss criterion.
        :type loss_criterion: nn.Module
        :return: A tuple containing the loss, accuracy, precision, recall, F1 score, and the report.
        :rtype: tuple
        '''
        model.eval()
        loss_values = []
        predicted_labels_list = []
        true_labels_list = []

        with torch.no_grad():
            for input_data, target_labels, masks in eval_data_loader:
                model_output = model(input_data)

                class_count = model_output.size(-1)
                model_output = torch.masked_select(model_output, masks.unsqueeze(-1)).view(-1, class_count)
                target_labels = torch.masked_select(target_labels, masks)

                loss = loss_criterion(model_output, target_labels)

                loss_values.append(loss.item())
                predicted_labels_list.append(torch.argmax(model_output, dim=-1).cpu().numpy())
                true_labels_list.append(target_labels.cpu().numpy())

        predicted_labels = np.concatenate(predicted_labels_list)
        true_labels = np.concatenate(true_labels_list)

        average_loss = np.mean(loss_values)
        evaluation_scores = calculate_evaluation_scores(true_labels, predicted_labels)

        return (average_loss, *evaluation_scores)
    

    def calculate_class_weights(self, targets: List[List[int]]) -> List[float]:
        '''
        Calculate the class weights for the dataset.
        :param target_labels: The target labels.
        :type target_labels: list
        :return: The class weights.
        :rtype: list
        '''
        # Flatten the list of lists
        flat_targets = [label for sublist in targets for label in sublist]

        # Count the frequency of each class
        class_counts = {}
        for label in flat_targets:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Calculate the total number of samples
        total_samples = len(flat_targets)

        # Calculate class weights
        class_weights = []
        for label, count in class_counts.items():
            weight = total_samples / (len(class_counts) * count)
            class_weights.append(weight)

        return class_weights
    

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

        
        if self.isTrain:
            if not os.path.exists(self.dev_preprocess):
                print("Preprocessing the validation data")
                inputs, targets = self.read_dataset_dailyDialog(
                    self.dev_data_path, self.dev_label_path)
                self.helper.dump_tuple(self.dev_preprocess, (inputs, targets))
            else:
                print("The preprocessed dev data already exists")


        if not os.path.exists(self.test_preprocess):
            print("Preprocessing the testing data")
            inputs, targets = self.read_dataset_dailyDialog(self.test_data_path, self.test_label_path)
            self.helper.dump_tuple(self.test_preprocess, (inputs, targets))
        else:
            print("The preprocessed test data already exists")

        
        if self.isTrain:
            # load data (train, dev, test)
            train_data, train_label = self.helper.load_tuple(self.train_preprocess)
            dev_data, dev_label = self.helper.load_tuple(self.dev_preprocess)
        test_data, test_label = self.helper.load_tuple(self.test_preprocess)



        if self.isTrain:
            # get the embedding of the data (Train, dev, test)
            train_embeddings = self.dataset_embedding(train_data, self.word2vec.word2vec)
            dev_embeddings = self.dataset_embedding(dev_data, self.word2vec.word2vec)

        test_embeddings = self.dataset_embedding(test_data, self.word2vec.word2vec)
        print("Embeddings obtained.")


        # Define the class weights
        class_weights = torch.tensor(self.calculate_class_weights(train_label)).to(self.DEVICE)

        # build dataset
        if self.isTrain:
            # train and dev data set
            train_dataset = EDCDataset(train_embeddings, train_label)
            dev_dataset = EDCDataset(dev_embeddings, dev_label)
            # train and dev data loader
            train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, collate_fn=collate_data, shuffle=True)
            dev_data_loader = DataLoader(dataset=dev_dataset, batch_size=self.batch_size, collate_fn=collate_data, shuffle=False)

        # test data set and data loader
        test_dataset = EDCDataset(test_embeddings, test_label)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, collate_fn=collate_data, shuffle=False)

        # build model
        model = Model(self.embedding_size, self.lstm_hidden_size, self.hidden_layer_size).to(self.DEVICE)

        # Use weighted cross-entropy loss
        loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), self.learning_rate)

        if self.isTrain:
            best_f1 = 0
            for i in range(self.epochs):
                print(f'Epoch: {i}')
                train_loss, train_acc, train_p, train_r, train_f1, _ = self.train_model(model, train_data_loader, loss_criterion, optimizer)
                dev_loss, dev_acc, dev_p, dev_r, dev_f1, _ = self.evaluate_model(model, dev_data_loader, loss_criterion)
                print("Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(train_loss, train_acc, train_f1, train_p, train_r))
                print("Dev   Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(dev_loss, dev_acc, dev_f1, dev_p, dev_r))
                if dev_f1 > best_f1:
                    print('Found Best :)')
                    best_f1 = dev_f1
                    torch.save(model.state_dict(), self.model_save_path)
                print('--------------------------------------------')

        model.load_state_dict(torch.load(self.model_save_path, map_location=self.DEVICE))
        model = model.to(self.DEVICE)

        test_loss, test_accuracy, test_percision, test_recall, test_f1, result_report = self.evaluate_model(model, test_data_loader, loss_criterion)
        print("Test   Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(test_loss, test_accuracy, test_f1, test_percision, test_recall))
        print(result_report)


    # preprocess single dialogue
    def preprocess(self, dialogue: List[str], clean: Callable[[str], List[str]], word2vec: Callable[[str], torch.Tensor]) -> torch.Tensor:
        '''
        Preprocess a single dialogue.
        :param dialogue: The dialogue to preprocess.
        :type dialogue: List[str]
        :param clean: The cleaning function.
        :type clean: Callable[[str], List[str]]
        :param word2vec: The word to vector function.
        :type word2vec: Callable[[str], torch.Tensor]
        :return: The preprocessed dialogue.
        :rtype: torch.Tensor
        '''
        # Tokenize each sentence into words and convert to embeddings
        conversation_tokens = []
        for sentence in dialogue:
            words = clean(sentence, steps=self.steps)
            tokens = [word2vec(word) for word in words]
            conversation_tokens.append(torch.stack(tokens).to(self.DEVICE))

        # Convert to the required format for input to the model
        packed_input = pack_sequence(conversation_tokens, enforce_sorted=False)
        return packed_input
    
    def emotion_detection(self, dialogue: List[str], model: nn.Module) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Predict the class of the dialogue using the model.

        Args:
            dialogue (List[str]): The dialogue, represented as a list of sentences.
            model (nn.Module): The model.

        Returns:
            Tuple[List[str], List[Dict[str, float]]]: The names of the classes with the highest probability for each sentence and a list of dictionaries mapping class names to probabilities for each sentence.
        """
        model.eval()

        # Preprocess the dialogue to get the input data
        input_data = self.preprocess(dialogue, self.preprocessor.clean, self.word2vec.word2vec)

        with torch.no_grad():
            output = model([input_data])

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=-1)

        # Get the class with the highest probability for each sentence
        predicted_classes = torch.argmax(probabilities, dim=-1)

        # Flatten the list of lists to a single list
        predicted_classes_flat = [i for sublist in predicted_classes.tolist() for i in sublist]

        # Map the class indices to the class names
        predicted_class_names = [self.emotion_map[i] for i in predicted_classes_flat]

        # Flatten the list of lists to a single list
        probabilities_flat = [prob for sublist in probabilities.tolist() for prob in sublist]

        # Map the probabilities to the class names for each sentence
        class_probabilities = [{self.emotion_map[i]: round((prob * 100), 3) for i, prob in enumerate(sentence)} for sentence in probabilities_flat]


        return predicted_class_names, class_probabilities
    
    
    
    def prod_mode(self):

        # Load the trained model
        model = Model(self.embedding_size, self.lstm_hidden_size, self.hidden_layer_size).to(self.DEVICE)
        model.load_state_dict(torch.load(self.model_save_path, map_location=self.DEVICE))
        model.eval()
        model.to(self.DEVICE)


        dialogue = ["Hello, how are you?", "I'm good, thanks! How about you?", "I'm doing well, thank you."]
        predicted_class_names, class_probabilities = self.emotion_detection(dialogue, model)
        for sentence, predicted_class_name, probs in zip(dialogue, predicted_class_names, class_probabilities):
            print(f"Sentence: {sentence}")
            print(f"Predicted class: {predicted_class_name}")
            print()
            print("Class probabilities:")
            for class_name, probability in probs.items():
                print(f"{class_name}: {probability}")
            print('----------------------------------------------')


print('finished loading class')

if __name__ == '__main__':
    while True:
        print('''
            Choose the mode you want to operate...
            1. Train Mode
            2. Production Mode
            else to exit
            ''')
        mode = input("Enter your choice: ")
        if mode == '1':
            emotion_model = Emotion(isTrain=True)
            emotion_model.dev_mode()
        elif mode == '2':
            emotion_model = Emotion()
            emotion_model.prod_mode()
        else:
            print('Exit...')
            exit()

