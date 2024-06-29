import random
import json
import pandas as pd
import joblib
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from constants import *
from utils import *

class chatbot_interfacing:
  def __init__(self, model_name) -> None:
    '''
    Load the dataset and initialize the model
    Args:
      model_name (str): The name of the model to be used
    '''
    self.dataset = None
    self.load_dataset()
    self.model_initialization(model_name=model_name)
    
  def load_dataset(self):
    '''
    Load the dataset
    '''
    
    with open(data_path, 'r') as f:
      data = json.load(f)

    df = pd.DataFrame(data['intents'])
    print(f'data set loaded successfully, dataset size = {len(df)}')
    self.dataset = map_tag_pattern(df, tag_col, text_col, res_col)

  # Function to predict the intent
  def predict_intent(self, user_input, vectorizer, model):
      '''
      Predict the intent of the user input
      Args:
        user_input (str): The user input
        vectorizer (TfidfVectorizer): The vectorizer object
        model (LogisticRegression or SVC): The model object
      Returns:
        intent (str): The intent detected
      '''
      user_input_vec = vectorizer.transform([user_input])
      intent = model.predict(user_input_vec)[0]
      return intent

  # Function to generate a random response based on the intent
  def generate_response(self, intent):
      '''
      Generate a random response based on the intent
      Args:
        intent (str): The intent detected
      Returns:
        response (str): The response to be given to the user
      '''
      # print(intent)  # For debugging purposes, to see the intent detected
      
      # Filter the DataFrame to get the responses for the given intent
      responses = self.dataset[self.dataset['tag'] == intent]['responses'].values[0]

      # Select a random response
      response = random.choice(responses)
      return response
  
  def model_initialization(self, model_name):
    '''
    Initialize the model
    Args:
      model_name (str): The name of the model to be used
    Returns:
      vectorizer (TfidfVectorizer): The vectorizer object
      model (LogisticRegression or SVC): The model object
    '''
    # Load the vectorizer and the model from local files
    vectorizer = joblib.load(vectorizer_filename)
    if model_name == lr_model_name:
        model = joblib.load(lr_model)
    elif model_name == svc_model_name:
        model = joblib.load(svc_model)
    return vectorizer, model
  
  def main(self): 
    '''
    Main function to run the chatbot
    '''
    vectorizer, model = self.model_initialization(model_name)
    while True:
        user_input = input("Prompt (press 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        else:
            intent = self.predict_intent(user_input, vectorizer, model)
            response = self.generate_response(intent)
            print("AI:", response)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intent Prediction Model')
    parser.add_argument('--model', type=str, required=True, help='svc or lr')
    args = parser.parse_args()
    model_name = args.model
    chatbot = chatbot_interfacing(model_name)
    chatbot.main()