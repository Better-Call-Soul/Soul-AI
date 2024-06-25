# Import Required Libraries
import json
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

from utils import *
from logistic_regression import LogisticRegressionModel
from svc import SVCModel

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
    with open('../../../data/processed/chatbot/input.json', 'r') as f:
      data = json.load(f)

    self.dataset = pd.DataFrame(data['intents'])
    print(f'data set loaded successfully, dataset size = {len(self.dataset)}')
    
  def map_dataset(self):
    df = map_tag_pattern(self.dataset, "tag", "patterns", "responses")
    X = df['patterns']
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test
      
  def initialize_model(self, model, get_best_params=False):
    if model == 'logistic_regression':
      self.model = LogisticRegressionModel(save_model=self.save_model, get_best_params = get_best_params)
    elif model == 'svc':
      self.model = SVCModel(save_model=self.save_model, get_best_params = get_best_params)

  def train(self):
    X_train_vec, X_test_vec, y_train, y_test = self.map_dataset()
    
    model = self.model.fit(X_train_vec, y_train)
    y_pred, accuracy_sv, F1_sv, recall_sv, precision_sv = self.model.predict(X_test_vec, y_test)

    print(f'Accuracy: {accuracy_sv}, F1: {F1_sv}, Recall: {recall_sv}, Precision: {precision_sv}')
