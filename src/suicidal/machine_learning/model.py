import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple
# from preprocess import Preprocessor
from logistic_regression_model import * 
from random_forest import *

class SuicidalDetectionDataset:
    def __init__(self, train_data_path: str, test_data_path: str):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.forward_label_mapping = {'suicide': 1, 'non-suicide': 0}
        self.reverse_label_mapping = {0: 'non-suicide', 1: 'suicide'}
        self.df = None
        # self.preprocessor = Preprocessor()
        # self.steps = [
        #         'translate_emojis_to_text',
        #         'lower_sentence',
        #         'remove_nonascii_diacritic',
        #         'remove_emails',
        #         'clean_html',
        #         'remove_url',
        #         'replace_repeated_chars',
        #         'expand_sentence',
        #         'remove_possessives',
        #         'remove_extra_space',
        #         'tokenize_sentence'
        #         ]

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

    # def clean_data(self, data: List[str]) -> List[List[str]]:
    #     '''
    #     Clean the data.
    #     :param data: The data to clean.
    #     :type data: List[str]
    #     :return: The cleaned data.
    #     :rtype: List[List[str]]
    #     '''
    #     cleaned_data = []
    #     for line in tqdm(data):
    #         cleaned_line = self.preprocessor.clean(line, steps=self.steps)
    #         cleaned_data.append(cleaned_line)
    #     return cleaned_data

# Example usage:
if __name__ == "__main__":
    train_dataset_path = 'data/processed/suicidal_detection_train.csv'
    test_dataset_path = 'data/processed/suicidal_detection_test.csv'
    dataset = SuicidalDetectionDataset(train_dataset_path,test_dataset_path)
    

    dialogues_train, labels_train  = dataset.read_dataset("train")
    dialogues_test, labels_test  = dataset.read_dataset("test")

    isTrain = True
    model_type = "rf"
    if model_type == "lr":
        model = LogisticRegressionModel()
    elif model_type == "rf":
        model = RandomForestModel()
    if isTrain:
        model.train_and_tune(dialogues_train,labels_train,dialogues_test,labels_test)
        model.save_model(f'models/{model_type}_best_suicidal_detection_model.joblib')
    else:
        model.load_model(f'models/{model_type}_best_suicidal_detection_model.joblib')
    
    test_sentence = "I'm feeling very down and hopeless."

    prediction = model.predict(test_sentence)
    print(f'Prediction for "{test_sentence}": {prediction}')

