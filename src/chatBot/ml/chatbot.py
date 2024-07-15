# Import Required Libraries
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

from utils import *
from constants import *
from logistic_regression import LogisticRegressionModel
from svc import SVCModel

import joblib

class Chatbot:
  def __init__(self, model, save_model=False):
    self.save_model = save_model
    self.model = None
    self.dataset = None
    
    # dataset loading 
    self.load_dataset()

    # initialize model
    self.initialize_model(model)
    
  def load_dataset(self):
    with open(data_path, 'r') as f:
      data = json.load(f)

    self.dataset = pd.DataFrame(data['intents'])
    print(f'data set loaded successfully, dataset size = {len(self.dataset)}')
    
  def map_dataset(self):
      '''
      Map the dataset to the required format for training
      Returns:
        X_train_vec: The vectorized training data 
        X_test_vec: The vectorized testing data
        y_train: The training labels
        y_test: The testing labels
      '''
      df = map_tag_pattern(self.dataset, tag_col, text_col, res_col)
      X = df[text_col]
      y = df[tag_col]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      vectorizer = TfidfVectorizer()
      X_train_vec = vectorizer.fit_transform(X_train)
      X_test_vec = vectorizer.transform(X_test)

      # Save the vectorizer to a file
      joblib.dump(vectorizer, vectorizer_filename)

      return X_train_vec, X_test_vec, y_train, y_test
      
  def initialize_model(self, model, get_best_params=False):
    '''
    Initialize the model
    Args:
      model (str): The model to initialize
      get_best_params (bool): Whether to get the best parameters for the model
    '''
    if model == lr_model_name:
      self.model = LogisticRegressionModel(save_model=self.save_model, get_best_params = get_best_params)
    elif model == svc_model_name:
      self.model = SVCModel(save_model=self.save_model, get_best_params = get_best_params)

  def train(self):
    '''
    Train the model
    '''
    X_train_vec, X_test_vec, y_train, y_test = self.map_dataset()
    
    self.model.fit(X_train_vec, y_train)
    y_pred, accuracy_sv, F1_sv, recall_sv, precision_sv = self.model.predict(X_test_vec, y_test)

    print(f'Accuracy: {accuracy_sv}, F1: {F1_sv}, Recall: {recall_sv}, Precision: {precision_sv}')
    
    # report_classification(y_test, y_pred, filename=classification_report)


if __name__ == '__main__':
  chatbot = Chatbot('lr', save_model=True)
  chatbot.train()