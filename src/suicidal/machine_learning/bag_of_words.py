from typing import List, Tuple
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

class BagOfWordsScratch:
    def __init__(self, max_features: int = None):
        '''
        Initializes a BagOfWordsScratch object.
        
        :param max_features: Maximum number of unique tokens to consider in the vocabulary.
                             If None, all unique tokens will be considered.
        :type max_features: int, optional
        '''
        self.vocabulary = {}
        self.max_features = max_features
    
    def fit(self, documents: List[str]):
        '''
        Builds the vocabulary from the input documents.
        
        :param documents: A list of text documents to build the vocabulary from.
        :type documents: List[str]
        '''
        vocab_counter = Counter()
        for doc in documents:
            words = self.tokenize(doc)
            vocab_counter.update(words)
        
        if self.max_features:
            most_common = vocab_counter.most_common(self.max_features)
            self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        else:
            self.vocabulary = {word: idx for idx, word in enumerate(vocab_counter.keys())}
    
    def transform(self, documents: List[str]) -> csr_matrix:
        '''
        Transforms the input documents into a sparse matrix of token counts based on the built vocabulary.
        
        :param documents: A list of text documents to transform.
        :type documents: List[str]
        :return: Sparse matrix of token counts.
        :rtype: csr_matrix
        '''
        rows, cols, data = [], [], []
        for row_idx, doc in enumerate(documents):
            word_counter = Counter(self.tokenize(doc))
            for word, count in word_counter.items():
                if word in self.vocabulary:
                    col_idx = self.vocabulary[word]
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(count)
        
        return csr_matrix((data, (rows, cols)), shape=(len(documents), len(self.vocabulary)))
    
    def fit_transform(self, documents: List[str]) -> csr_matrix:
        '''
        Fits the model to the input documents and transforms them into a sparse matrix of token counts.
        
        :param documents: A list of text documents to fit and transform.
        :type documents: List[str]
        :return: Sparse matrix of token counts.
        :rtype: csr_matrix
        '''
        self.fit(documents)
        return self.transform(documents)
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        '''
        Tokenizes the input text into a list of lowercase words.
        
        :param text: The text to tokenize.
        :type text: str
        :return: List of lowercase words.
        :rtype: List[str]
        '''
        return text.lower().split()