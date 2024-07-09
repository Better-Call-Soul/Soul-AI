import numpy as np
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# TFIDFVectorizer
# This class is used to vectorize text data using the TF-IDF algorithm
class TFIDFVectorizer:
    def __init__(self):
        # initialize the tokenizer, term frequency, inverse document frequency, and unique words
        self.idf_dict = defaultdict(int)
        self.unique_words = []
    
    # fit the vectorizer to the data
    def fit(self, statements):
        # tokenize the statements
        self.tokenized_statements = [self.tokenizer(statement) for statement in statements]
        # calculate the term frequency 
        self.tf = self.term_frequency(self.tokenized_statements)
        # calculate the inverse document frequency
        self.idf = self.inverse_document_frequency(self.tokenized_statements)
        # collect all words from the tokenized statements and create a sorted list of unique words
        self.unique_words = sorted(
            set(  # use a set to keep only unique words
                word  # the word to add to the set
                for statement in self.tokenized_statements  # iterate through each tokenized statement
                for word in statement  # iterate through each word in each statement
            )
        )
    
    # transform the data
    def transform(self, statements):
        # calculate the tf-idf matrix
        tf_idf_matrix = self.tf_idf(self.tf, self.idf)
        tf_idf_array = np.zeros((len(statements), len(self.unique_words)))
        # fill the numpy array of the tf-idf matrix
        for i, statement in enumerate(tf_idf_matrix):
            for word, val in statement.items():
                if word in self.unique_words:
                    word_index = self.unique_words.index(word)
                    tf_idf_array[i, word_index] = val

        return tf_idf_array

    # fit and transform the data
    def fit_transform(self, statements):
        self.fit(statements)
        return self.transform(statements)

    # tokenize the statements
    def tokenizer(self, statement):
        words = word_tokenize(statement)
        # remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]
        return words
    
    # calculate the term frequency
    def term_frequency(self, tokenized_statements):
        tf_array = []
        # loop through each statement
        for statement in tokenized_statements:
            # create a dictionary to store the term frequency of each word
            tf_dict = defaultdict(int)
            for word in statement:
                tf_dict[word] += 1 / len(statement)
            # append the dictionary to the array
            tf_array.append(tf_dict)
        return tf_array

    # calculate the inverse document frequency
    def inverse_document_frequency(self, tokenized_statements):
        # calculate the total number of statements
        total_statements = len(tokenized_statements)
        idf_dict = defaultdict(int)
        # loop through each statement
        for statement in tokenized_statements:
            # loop through each word in the statement
            for word in set(statement):
                idf_dict[word] += 1
        # calculate the inverse document frequency
        for word, count in idf_dict.items():
            idf_dict[word] = math.log(total_statements / (count + 1))
        return idf_dict

    # calculate the tf-idf matrix
    def tf_idf(self, tf, idf):
        # create an array to store the tf-idf matrix
        tf_idf_matrix = []
        # loop through each statement
        for statement in tf:
            # create a dictionary to store the tf-idf of each word
            doc_tf_idf = {}
            # loop through each word in the statement
            for word, freq in statement.items():
                # calculate the tf-idf of the word
                doc_tf_idf[word] = freq * idf[word]
            # append the dictionary to the array
            tf_idf_matrix.append(doc_tf_idf)
        return tf_idf_matrix
    
    # get the feature names
    def get_feature_names_out(self):
        return self.unique_words