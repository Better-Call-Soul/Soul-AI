import string
import pickle
import random
import json 
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


from constants import *
from utils import *

class Chatbot:
  def __init__(self):
      '''
      Load the tokenizer, label encoder and model
      '''
      # Load Tokenizer
      self.tokenizer = joblib.load(tokenizer_path)

      # Load LabelEncoder
      self.label_encoder = joblib.load(label_encoder_path)
          
      # Load Model
      with open(model_path, 'rb') as file:
          self.model = pickle.load(file)
          
      self.load_dataset()


  def load_dataset(self):
      '''
      Load the dataset and responses
      '''
      with open(data_path, 'r') as f:
        data = json.load(f)

      df = pd.DataFrame(data['intents'])

      _, _, self.responses = map_tag_pattern(df, tag_col, text_col, res_col)


  def predict(self, user_input):
      '''
      Predict the response for the user input
      :param user_input: The user input
      :type user_input: str
      '''
      textList = []
      prediction_input = []
      for letter in user_input:
          if letter not in string.punctuation:
              prediction_input.append(letter.lower())

      prediction_input = ''.join(prediction_input)
      textList.append(prediction_input)

      prediction_input = self.tokenizer.texts_to_sequences(textList)
      prediction_input = np.array(prediction_input).reshape(-1)
      prediction_input = pad_sequences([prediction_input], 18)

      output = self.model.predict(prediction_input)
      output = output.argmax()

      response_tag = self.label_encoder.inverse_transform([output])[0]
      print("AI: ", random.choice(self.responses[response_tag]))



chatbot = Chatbot()

while True:
  user_input = input("Input (press 'q' to quit): ")
  if user_input.lower() == 'q':
      break
  else:
      chatbot.predict(user_input)
