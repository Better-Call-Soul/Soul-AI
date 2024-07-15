from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import json
import random
import pandas as pd
import numpy as np
import pickle
import joblib

from utils import *
from constants import *
from constants import data_path, text_col, res_col

import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Now you can import Preprocessor from preprocess.py
from preprocess.preprocess import Preprocessor

class Chatbot:
  def __init__(self, save_model=False):
    self.save_model = save_model
    self.model = None
    self.tokenizer = None
    
    self.train_data = None
    self.train_labels = None
    
    self.encoded_labels = None
    self.train_sequences = None

    self.preprocessor = Preprocessor()    

    # dataset loading 
    self.load_dataset()

    # initialize model
    self.initialize_model()
    
  def load_dataset(self):
    with open(data_path, 'r') as f:
      data = json.load(f)

    dataset = pd.DataFrame(data['intents'])
    # print(f'data set loaded successfully, dataset size = {len(dataset)}')
    
    self.train_data, self.train_labels = map_tag_pattern(self.preprocessor, dataset, text_col, res_col)
    
    # for item in zip(train_data, train_labels):
    #   print(item)

  def initialize_model(self):
    # Encoding the labels using LabelEncoder
    label_encoder = LabelEncoder()
    self.encoded_labels = label_encoder.fit_transform(self.train_labels)

    # Tokenizing the training data
    self.tokenizer = keras.preprocessing.text.Tokenizer()
    self.tokenizer.fit_on_texts(self.train_data)
    self.train_sequences = self.tokenizer.texts_to_sequences(self.train_data)
    self.train_sequences = keras.preprocessing.sequence.pad_sequences(self.train_sequences)
    
    with open('tokenizer.pickle', 'wb') as handle:
      pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('train_sequences.pickle', 'wb') as handle:
      pickle.dump(self.train_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Defining the Sequential model
    model = keras.models.Sequential()

    # Adding an Embedding layer
    model.add(keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, 
                                    output_dim=100, 
                                    input_length=self.train_sequences.shape[1]))

    # Adding a Flatten layer
    model.add(keras.layers.Flatten())

    # Adding a Dense layer with ReLU activation
    model.add(keras.layers.Dense(64, activation='relu'))

    # Adding the output layer with softmax activation
    model.add(keras.layers.Dense(len(np.unique(self.encoded_labels)), activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    self.model = model
    
  def train(self, epochs=100, batch_size=8):
    self.model.fit(self.train_sequences, self.encoded_labels, epochs=epochs, batch_size=batch_size)
    
    if self.save_model:
      self.model.save(model_path)
    
  

if __name__ == '__main__':
  chatbot = Chatbot(save_model=True)
  chatbot.train(epochs=100, batch_size=8)
  # print(chatbot.generate_response('Hi'))
  

