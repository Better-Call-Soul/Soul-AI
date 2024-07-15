
import string
import pickle
import random
import json 
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


from constants import *
from utils import *

import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Now you can import Preprocessor from preprocess.py
from preprocess.preprocess import Preprocessor


class Chatbot:
    def __init__(self):
        '''
        Load the tokenizer, label encoder and model
        '''
        # Load Model
        self.model = load_model(model_path)
            
        self.preprocessor = Preprocessor()    
        
        self.load_dataset()

        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open('train_sequences.pickle', 'rb') as handle:
            self.train_sequences = pickle.load(handle)

        self.label_encoder = joblib.load('label_encoder.pkl')

    def load_dataset(self):
        with open(data_path, 'r') as f:
           data = json.load(f)

        dataset = pd.DataFrame(data['intents'])
        # print(f'data set loaded successfully, dataset size = {len(dataset)}')
        
        self.train_data, self.train_labels = map_tag_pattern(self.preprocessor, dataset, text_col, res_col)
        
        # for item in zip(train_data, train_labels):
        #   print(item)
        
    # Function to generate response based on the input text
    def generate_response(self, text):
        # Tokenizing and padding the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        sequence = pad_sequences(sequence, maxlen=self.train_sequences.shape[1])
        
        # Making a prediction
        prediction = self.model.predict(sequence)
        
        # Getting the label with the highest predicted probability
        predicted_label = np.argmax(prediction)
        
        # Decoding the predicted label
        response = self.label_encoder.inverse_transform([predicted_label])[0]
        
        return response


chatbot = Chatbot()

# Running an interactive loop for user input
while True:
    user_input = str(input("Input: (press 'q' to quit) "))
    
    if user_input.lower() == "q":
        print("Response: Exiting.....")
        break

    # Assuming `preprocessor.clean` is a predefined function to clean the user input
    cleaned_input = chatbot.preprocessor.clean(user_input, preprocessing_steps, '')[0]
    
    # Generating and printing the response
    response = chatbot.generate_response(cleaned_input)
    print("Response:", response)