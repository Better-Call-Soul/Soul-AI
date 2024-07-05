from typing import List, Tuple, Callable, Dict
import random
import torch.autograd
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
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


class Suicidal:
    def __init__(self,
                dataset_path='../../data/raw/suicidal_detection',
                model_path='../../models/suicidal',
                isTrain=False,
                split_seed=10,
                seed=2,
                model_name='model_1E_test',
                batch_size=64,
                embedding_size=300,
                lstm_hidden_size=20,
                hidden_layer_size=256,
                cnn_layer_size=1,
                class_layer=1,
                classification_threshold=0.4,
                learning_rate=0.1,
                epochs=1,
                num_classes=2,
                momentum=0.09,
                word2vec=Glove(),
                ):
        # The folder of the data set
        self.dataset_path = dataset_path
        # The folder for the models
        self.model_path = model_path
        # Variables definition
        self.isTrain = isTrain
        self.split_seed = split_seed
        self.seed = seed

        self.model_name = model_name
        self.data_path = f"{self.dataset_path}/Suicide_Detection_Reddit.csv"
        self.train_preprocess = f"{self.dataset_path}/train/dialogues_train_preprocess.pkl"
        self.dev_preprocess = f"{self.dataset_path}/validation/dialogues_validation_preprocess.pkl"
        self.test_preprocess = f"{self.dataset_path}/test/dialogues_test_preprocess.pkl"
        self.model_save_path = f"{self.model_path}/{self.model_name}.pt"

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.cnn_layer_size = cnn_layer_size
        self.class_layer = class_layer
        self.classification_threshold = classification_threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes
        self.momentum = momentum

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define the mapping from 's' to 1 and 'u' to 0
        self.forward_label_mapping = {'suicide': 1, 'non-suicide': 0}

        # define the reverse mapping from 0 to 'u' and 1 to 's'
        self.reverse_label_mapping = {0: 'non-suicide', 1: 'suicide'}

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

    def read_dataset_suicidal_detection(self, data_path: str, split_seed: int) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
        '''
        Read the dataset for the suicidal detection task.
        :param data_path: The path to the dataset.
        :type data_path: str
        :param split_seed: The seed to use for splitting the dataset.
        :type split_seed: int
        :return: The training, validation, and test sets.
        :rtype: Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]
        '''
        data = pd.read_csv(data_path)
        # extract the dialogues and their corresponding labels
        dialogues = data['text'].tolist()
        labels = data['class'].apply(lambda x: self.forward_label_mapping[x]).tolist()

        # split the data into training and temporary sets (70% training, 30% temporary)
        dialogues_train, dialogues_temp, labels_train, labels_temp = train_test_split(
            dialogues, labels, test_size=0.3, random_state=split_seed)

        # split the temporary set into validation and test sets (66.67% validation, 33.33% test)
        dialogues_val, dialogues_test, labels_val, labels_test = train_test_split(
            dialogues_temp, labels_temp, test_size=0.33, random_state=split_seed)

        return (dialogues_train, labels_train), (dialogues_val, labels_val), (dialogues_test, labels_test)
    

    def clean_data(self, data: List[str]) -> List[List[str]]:
        '''
        Clean the data.
        :param data: The data to clean.
        :type data: List[str]
        :return: The cleaned data.
        :rtype: List[List[str]]
        '''
        cleaned_data = []
        for line in tqdm(data):
            cleaned_line = self.preprocessor.clean(line, steps=self.steps)
            cleaned_data.append(cleaned_line)
        return cleaned_data



    def train_model(self, model: nn.Module, train_data_loader: DataLoader, loss_criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[float, float, float, float, float, str]:
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

        for input_data, target_labels in train_data_loader:

            # Move data and labels to device
            input_data = input_data.to(self.DEVICE)
            target_labels = target_labels.to(self.DEVICE).float()

            optimizer.zero_grad()
            model_output = model(input_data)

            # Flatten model_output
            model_output_flat = model_output.view(-1)
            target_labels_flat = target_labels.view(-1)

            # Calculate loss
            loss = loss_criterion(model_output_flat, target_labels_flat)

            # Backpropagation
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            predicted_labels_list.append(
                (model_output_flat > self.classification_threshold).float().cpu().numpy())
            true_labels_list.append(target_labels_flat.cpu().numpy())

        predicted_labels = np.concatenate(predicted_labels_list)
        true_labels = np.concatenate(true_labels_list)

        average_loss = np.mean(loss_values)
        evaluation_scores = calculate_evaluation_scores(
            true_labels, predicted_labels)

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
            for input_data, target_labels in eval_data_loader:

                # Move data and labels to device
                input_data = input_data.to(self.DEVICE)
                target_labels = target_labels.to(self.DEVICE).float()

                model_output = model(input_data)

                # Flatten model_output
                model_output_flat = model_output.view(-1)
                target_labels_flat = target_labels.view(-1)

                loss = loss_criterion(model_output_flat, target_labels_flat)

                loss_values.append(loss.item())
                predicted_labels_list.append(
                    (model_output_flat > self.classification_threshold).float().cpu().numpy())
                true_labels_list.append(target_labels_flat.cpu().numpy())

        predicted_labels = np.concatenate(predicted_labels_list)
        true_labels = np.concatenate(true_labels_list)

        average_loss = np.mean(loss_values)
        evaluation_scores = calculate_evaluation_scores(
            true_labels, predicted_labels)

        return (average_loss, *evaluation_scores)
    

    def test_model(self, model: nn.Module, eval_data_loader: DataLoader, loss_criterion: nn.Module) -> Tuple[float, float, float, float, float, str]:
        '''
        Test the model using the test data.
        :param model: The model to evaluate.
        :type model: nn.Module
        :param eval_data_loader: The DataLoader for the evaluation data.
        :type eval_data_loader: DataLoader
        :param loss_criterion: The loss criterion.
        :type loss_criterion: nn.Module
        :return: A tuple containing the loss, accuracy, precision, recall, F1 score, percision recall threshould and the report.
        :rtype: tuple
        '''
        model.eval()
        loss_values = []
        predicted_labels_list = []
        true_labels_list = []
        prob_list = []

        with torch.no_grad():
            for input_data, target_labels in eval_data_loader:

                # Move data and labels to device
                input_data = input_data.to(self.DEVICE)
                target_labels = target_labels.to(self.DEVICE).float()

                model_output = model(input_data)

                # Flatten model_output
                model_output_flat = model_output.view(-1)
                target_labels_flat = target_labels.view(-1)

                loss = loss_criterion(model_output_flat, target_labels_flat)

                loss_values.append(loss.item())
                predicted_labels_list.append(
                    (model_output_flat > self.classification_threshold).float().cpu().numpy())
                true_labels_list.append(target_labels_flat.cpu().numpy())

                prob_list.append(model_output_flat.float().cpu().numpy())

        predicted_labels = np.concatenate(predicted_labels_list)
        true_labels = np.concatenate(true_labels_list)

        average_loss = np.mean(loss_values)
        evaluation_scores = calculate_evaluation_scores(
            true_labels, predicted_labels)

        precision, recall, thresholds = precision_recall_curve(true_labels, predicted_labels)

        fpr, tpr, thresholds_2 = roc_curve(true_labels, np.concatenate(prob_list))
        roc_au_val = roc_auc_score(true_labels, np.concatenate(prob_list))

        return (average_loss, *evaluation_scores, precision, recall, thresholds, fpr, tpr, thresholds_2, roc_au_val)
    
    def dev_mode(self):
        print(f'starting dev mode with train is set to {self.isTrain}')
        # read the data set
        (dialogues_train, labels_train), (dialogues_val, labels_val), (dialogues_test,
                                                                    labels_test) = self.read_dataset_suicidal_detection(self.data_path, self.split_seed)
        if self.isTrain:
            if not os.path.exists(self.train_preprocess):
                print("Preprocessing the training data")
                dialogues_train_proccessed = self.clean_data(dialogues_train)
                self.helper.dump_tuple(self.train_preprocess,(dialogues_train_proccessed, labels_train))
            else:
                print("The preprocessed train data already exists")
                dialogues_train_proccessed, labels_train = self.helper.load_tuple(self.train_preprocess)
        

        if self.isTrain:
            if not os.path.exists(self.dev_preprocess):
                print("Preprocessing the validation data")
                dialogues_val_proccessed = self.clean_data(dialogues_val)
                self.helper.dump_tuple(self.dev_preprocess, (dialogues_val_proccessed, labels_val))
            else:
                print("The preprocessed dev data already exists")
                dialogues_val_proccessed, labels_val = self.helper.load_tuple(self.dev_preprocess)

        if not os.path.exists(self.test_preprocess):
            print("Preprocessing the testing data")
            dialogues_test_proccessed = self.clean_data(dialogues_test)
            self.helper.dump_tuple(self.test_preprocess, (dialogues_test_proccessed, labels_test))
        else:
            print("The preprocessed test data already exists")
            dialogues_test_proccessed, labels_test = self.helper.load_tuple(self.test_preprocess)
        
        # build dataset
        if self.isTrain:
            # train and dev data set
            train_dataset = SUIDataset(dialogues_train_proccessed, labels_train, self.word2vec.word2vec)
            dev_dataset = SUIDataset(dialogues_val_proccessed, labels_val, self.word2vec.word2vec)
            # train and dev data loader
            train_data_loader = DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
            dev_data_loader = DataLoader(
                dataset=dev_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)

        # test data set and data loader
        test_dataset = SUIDataset(dialogues_test_proccessed, labels_test, self.word2vec.word2vec)
        test_data_loader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)

        # build model
        model = Model(self.embedding_size, self.lstm_hidden_size, self.hidden_layer_size,
                    self.cnn_layer_size, self.class_layer).to(self.DEVICE)

        # Use binary cross-entropy loss
        # loss_criterion = nn.BCELoss()
        # loss_criterion = nn.BCEWithLogitsLoss()
        loss_criterion = nn.BCELoss()

        # optimizer = optim.Adam(model.parameters(), learning_rate, momentum=momentum)
        # Define the optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        if self.isTrain:
            best_f1 = 0
            for i in range(self.epochs):
                print(f'Epoch: {i}')
                train_loss, train_acc, train_p, train_r, train_f1, _ = self.train_model(
                    model, train_data_loader, loss_criterion, optimizer)
                dev_loss, dev_acc, dev_p, dev_r, dev_f1, _ = self.evaluate_model(
                    model, dev_data_loader, loss_criterion)
                print("Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(
                    train_loss, train_acc, train_f1, train_p, train_r))
                print("Dev   Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(
                    dev_loss, dev_acc, dev_f1, dev_p, dev_r))
                if dev_f1 > best_f1:
                    print('Found Best :)')
                    best_f1 = dev_f1
                    torch.save(model.state_dict(), self.model_save_path)
                print('--------------------------------------------')

        model.load_state_dict(torch.load(self.model_save_path, map_location=self.DEVICE))
        model = model.to(self.DEVICE)

        test_loss, test_accuracy, test_percision, test_recall, test_f1, result_report, _, _, _, _, _, _, _ = self.test_model(model, test_data_loader, loss_criterion)
        print("Test   Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(
            test_loss, test_accuracy, test_f1, test_percision, test_recall))
        print(result_report)

    def preprocess(self, sentence: str, clean: Callable[[str], List[str]], word2vec: Callable[[str], torch.Tensor]) -> torch.Tensor:
        '''
        Preprocess a single sentence.
        :param sentence: The sentence to preprocess.
        :type sentence: str
        :param clean: The cleaning function.
        :type clean: Callable[[str], List[str]]
        :param word2vec: The word to vector function.
        :type word2vec: Callable[[str], torch.Tensor]
        :return: The preprocessed sentence.
        :rtype: torch.Tensor
        '''
        # Tokenize the sentence into words and convert to embeddings
        words = clean(sentence, steps=self.steps)
        sentence_tokens = [word2vec(word) for word in words]

        # Convert to the required format for input to the model
        sentence_tensor = torch.stack(sentence_tokens).to(self.DEVICE)

        return sentence_tensor
    
    def suicidal_detection(self, sentence: str, model: nn.Module) -> Tuple[str, Dict[str, float]]:
        """
        Predict the class of the sentence using the model.

        Args:
            sentence (str): The sentence.
            model (nn.Module): The model.

        Returns:
            Tuple[str, Dict[str, float]]: The name of the class with the highest probability and a dictionary mapping class names to probabilities.
        """
        model.eval()

        # Preprocess the sentence to get the input data
        input_data = self.preprocess(sentence, self.preprocessor.clean, self.word2vec.word2vec)

        with torch.no_grad():
            output = model(input_data.unsqueeze(0))

        # Apply sigmoid to get probabilities
        probabilities = output
        print(probabilities)

        # Threshold probabilities at 0.5 to make binary predictions
        predicted_class = (probabilities > self.classification_threshold).float()

        # Map the class index to the class name
        predicted_class_name = self.reverse_label_mapping[int(predicted_class.item())]

        # Map the probabilities to the class names
        sui_prob = probabilities.tolist()[0][0]
        sui_prob = round((sui_prob * 100), 3)
        non_prob = 100 - sui_prob
        class_probabilities = {
            self.reverse_label_mapping[0]:  non_prob,
            self.reverse_label_mapping[1]:  sui_prob
            }

        return predicted_class_name, class_probabilities
    
    def prod_mode(self):
        # Load the trained model
        model = Model(self.embedding_size, self.lstm_hidden_size, self.hidden_layer_size,
                    self.cnn_layer_size, self.class_layer)
        model.load_state_dict(torch.load(self.model_save_path, map_location=self.DEVICE))
        model.eval()
        model.to(self.DEVICE)
        model = model.to(self.DEVICE)

        dialogue = [
        "I will kill myself.",
        "I will not kill myself.",
        "I will not kill myself doing are unlucky investigating.",
        "I will kill myself doing are unlucky investigating.",
        "I will suicide.",
        "I will not suicide.",
        "I am sad.",
        "I am not sad..",
        "I am depressed.",
        "I am not depressed.",
        "I hate my team.",
        "I do not want to suicide.",
        "I want to suicide.",
        ]
        for sentence in dialogue:
            predicted_class_name, class_probabilities = self.suicidal_detection(
                sentence, model)
            print(f"Sentence: {sentence}")
            print(f"Predicted class: {predicted_class_name}")
            print()
            print("Class probabilities:")
            for class_name, probability in class_probabilities.items():
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
            suicidal_model = Suicidal(isTrain=True)
            suicidal_model.dev_mode()
        elif mode == '2':
            suicidal_model = Suicidal()
            suicidal_model.prod_mode()
        else:
            print('Exit...')
            exit()