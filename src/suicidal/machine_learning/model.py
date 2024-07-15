import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from quick_processor import Preprocessor
from logistic_regression_model import * 
from random_forest import *
from lr_bow import *
from tqdm import tqdm

class SuicidalDetectionDataset:
    def __init__(self, train_data_path: str, test_data_path: str):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.forward_label_mapping = {'suicide': 1, 'non-suicide': 0}
        self.reverse_label_mapping = {0: 'non-suicide', 1: 'suicide'}
        self.df = None
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

    def load_dataset(self,df_type: str):
        if df_type == "train":
            self.df = pd.read_csv(self.train_data_path)
        else:
            self.df = pd.read_csv(self.test_data_path)
    
    def read_dataset(self,df_type: str):
        self.load_dataset(df_type)
        
        dialogues = self.df['text'].tolist()
        labels = self.df['class'].apply(lambda x: self.forward_label_mapping[x]).tolist()
        
        return dialogues, labels
    def read_dataset_suicidal_detection(self,data_path: str, split_seed: int) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
        '''
        Read the dataset for the suicidal detection task.
        :param data_path: The path to the dataset.
        :type data_path: str
        :param split_seed: The seed to use for splitting the dataset.
        :type split_seed: int
        :return: The training, and test sets.
        :rtype: Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]
        '''
        data = pd.read_csv(data_path)
        # extract the dialogues and their corresponding labels
        dialogues = data['text'].tolist()
        labels = data['class'].apply(lambda x: self.forward_label_mapping[x]).tolist()

        # split the data into training and temporary sets (70% training, 30% temporary)
        dialogues_train, dialogues_test, labels_train, labels_test = train_test_split(
            dialogues, labels, test_size=0.3, random_state=split_seed)
        

        return (dialogues_train, labels_train),  (dialogues_test, labels_test)

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

# Example usage:
if __name__ == "__main__":
    # train_dataset_path = 'data/raw/suicidal_detection/Suicide_Detection.csv'
    # test_dataset_path = 'data/processed/suicidal_detection_test.csv'
    # dataset = SuicidalDetectionDataset(train_dataset_path,test_dataset_path)
    
    # (dialogues_train, labels_train),  (dialogues_test,labels_test) = dataset.read_dataset_suicidal_detection(train_dataset_path,10)

    # dialogues_train = dataset.clean_data(dialogues_train)
    # dialogues_test = dataset.clean_data(dialogues_test)
    # flattened_X_train = [' '.join(tokens) for tokens in dialogues_train]
    # flattened_X_test = [' '.join(tokens) for tokens in dialogues_test]

    # dialogues_train, labels_train  = dataset.read_dataset("train")
    # dialogues_test, labels_test  = dataset.read_dataset("test")

    isTrain = False
    model_type = "lr_bow"
    if model_type == "lr":
        model = LogisticRegressionModel()
    elif model_type == "rf":
        model = RandomForestModel()
    else:
        model = LogisticRegressionBowModel(BagOfWords(max_features=1000))
    if isTrain:
        model.train_and_tune(flattened_X_train,labels_train,flattened_X_test,labels_test)
        model.save_model(f'models/{model_type}_best_suicidal_detection_model.joblib')
    else:
        model.load_model(f'models/{model_type}_best_suicidal_detection_model.joblib')
    
    preprocessor = Preprocessor()
    
    test_sentence = "I'm feeling very down and hopeless."
    cleaned = preprocessor.clean(test_sentence)
    prediction = model.predict(" ".join(cleaned))
    print(f'Prediction for "{test_sentence}": {prediction}')

