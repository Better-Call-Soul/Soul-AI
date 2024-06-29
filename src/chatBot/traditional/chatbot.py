import numpy as np
import json
import random

from utils import *
from constants import *

import warnings
warnings.filterwarnings('ignore')

class Chatbot:
    def __init__(self):
        self.dataset = None
        self.load_dataset()
        
    def load_dataset(self):
        with open(data_path, 'r') as f:
          data = json.load(f)

        df = pd.DataFrame(data['intents'])

        self.dataset = map_tag_pattern(df, tag_col, text_col, res_col)

    def respond(self, text):
        '''
        Respond to the user input
        :param text: The user input
        :type text: str
        :return: The response to the user input
        :rtype: str
        '''
        maximum = float('-inf')
        responses = []
        for i in self.dataset.iterrows():
            sim = cosine_distance_countvectorizer_method(text, i[1][text_col])
            if sim > maximum:
                maximum = sim
                responses = i[1][res_col]
                response = random.choice(responses)
                closest = i[1][text_col]
        return response

if __name__ == "__main__":
    
    chatbot = Chatbot()

    while True:
        text = str(input("Input: (press 'q' to quit) "))
        if text.lower() == "q":
            print("Response: Exiting.....")
            break
        print("Response:", chatbot.respond(text))
