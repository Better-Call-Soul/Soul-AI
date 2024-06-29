import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

from utils import *
from constants import *

import warnings
warnings.filterwarnings('ignore')

class Chatbot_NN:
    def __init__(self):
        '''
        Load the dataset, initialize the training set, define hyperparameters and model
        '''
        self.tags = None
        self.inputs = None
        self.dataset = None
        self.load_dataset()
        self.initialize_trainingset()
        self.define_hyperparameters()
        self.define_model()
        
    def load_dataset(self):
        '''
        Load the dataset and responses
        '''
        with open(data_path, 'r') as f:
          data = json.load(f)

        df = pd.DataFrame(data['intents'])

        self.tags, self.inputs, _ = map_tag_pattern(df, tag_col, text_col, res_col)
        self.dataset = pd.DataFrame({"inputs":self.inputs, "tags":self.tags})
        self.dataset["inputs"] = self.dataset["inputs"].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))

    def initialize_trainingset(self):
        '''
        Initialize the training set
        '''
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.dataset["inputs"])
        self.encoded_inputs = self.tokenizer.texts_to_sequences(self.dataset["inputs"])
        # self.vocab_size = len(self.tokenizer.word_index) + 1
        # self.max_len = max([len(i.split()) for i in self.inputs])
        # self.padded_inputs = pad_sequences(self.encoded_inputs, maxlen=self.max_len, padding='post')
        self.encoded_inputs = pad_sequences(self.encoded_inputs)
        
        self.label_encoder = LabelEncoder()
        self.encoded_tags = self.label_encoder.fit_transform(self.dataset["tags"])

        # Save Tokenizer
        joblib.dump(self.tokenizer, tokenizer_path)

        # Save LabelEncoder
        joblib.dump(self.label_encoder, label_encoder_path)

    def define_hyperparameters(self):
        '''
        Define the hyperparameters
        '''
        self.epochs = 500
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.input_shape = self.encoded_inputs.shape[1] # 18
        self.embedding_dim = 10 # 128
        self.num_classes = self.label_encoder.classes_.shape[0]

    def define_model(self):
        '''
        Define the model
        '''
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.input_shape,)),
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=(self.input_shape,)),
            tf.keras.layers.LSTM(units=10, return_sequences=True),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation='relu'),
            tf.keras.layers.Dense(units=5, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()


    def train(self):
        '''
        Train the model
        '''
        # early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

        # train the model
        # self.model.fit(self.encoded_inputs, self.encoded_tags, epochs=self.epochs, callbacks=[early_stop])
        self.model.fit(self.encoded_inputs, self.encoded_tags, epochs=self.epochs)

        # Save Model
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)


chatbot = Chatbot_NN()
chatbot.train()