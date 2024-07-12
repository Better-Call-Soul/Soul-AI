from typing import List, Set, Dict, Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import unittest

# Ensure required NLTK data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

class CountVectorizer:
    def __init__(self, stop_words: Union[str, List[str]] = [], language: str = 'english'):
        """
        Initialize the CountVectorizer with optional stop words and language.
        """
        self.vocabulary_: Dict[str, int] = {}
        if stop_words == 'english':
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set(stop_words)

        self.word_tokenize = word_tokenize

    def get_vocabulary(self) -> Dict[str, int]:
        """
        Return the vocabulary as a dictionary.
        """
        return self.vocabulary_

    def get_stop_words(self) -> Set[str]:
        """
        Return the stop words as a set.
        """
        return self.stop_words
    
    def set_vocabulary(self, vocabulary: Dict[str, int]) -> None:
        """
        Set the vocabulary to a custom dictionary.
        """
        self.vocabulary_ = vocabulary

    def set_stop_words(self, stop_words: Union[Set[str], List[str]]) -> None:
        """
        Set the stop words to a custom list or set.
        """
        self.stop_words = set(stop_words)

    def fit(self, documents: List[str]) -> None:
        """
        Learn the vocabulary from the documents.
        """
        word_index = 0
        for document in documents:
            tokens = self.word_tokenize(document)
            for token in tokens:
                token = token.lower()
                if token not in self.stop_words and token.isalpha():
                    if token not in self.vocabulary_:
                        self.vocabulary_[token] = word_index
                        word_index += 1

    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transform documents to a document-term matrix.
        """
        matrix = [[0] * len(self.vocabulary_) for _ in documents]
        for doc_index, document in enumerate(documents):
            tokens = self.word_tokenize(document)
            for token in tokens:
                token = token.lower()
                if token in self.vocabulary_:
                    matrix[doc_index][self.vocabulary_[token]] += 1
        return matrix

    def fit_transform(self, documents: List[str]) -> List[List[int]]:
        """
        Fit the model and then transform the documents.
        """
        self.fit(documents)
        return self.transform(documents)

    def inverse_transform(self, matrix: List[List[int]]) -> List[str]:
        """
        Transform a document-term matrix back to documents.
        """
        inverse_vocab = {index: token for token, index in self.vocabulary_.items()}
        documents = []
        for row in matrix:
            document = []
            for index, count in enumerate(row):
                if count > 0:
                    document.extend([inverse_vocab[index]] * int(count))
            documents.append(' '.join(document))
        return documents

# Unit Testing
class TestCountVectorizer(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "The quick brown fox jumps over the lazy dog.",
            "The dog barks at the mailman.",
            "The cat sleeps on the windowsill."
        ]
        self.vectorizer = CountVectorizer()

    def test_fit_transform(self):
        X = self.vectorizer.fit_transform(self.documents)
        self.assertTrue(isinstance(X, List) and all(isinstance(row, List) for row in X))
        vocab = self.vectorizer.get_vocabulary()
        self.assertIn('quick', vocab)
        self.assertIn('dog', vocab)

    def test_inverse_transform(self):
        X = self.vectorizer.fit_transform(self.documents)
        documents_reconstructed = self.vectorizer.inverse_transform(X)
        self.assertEqual(len(documents_reconstructed), len(self.documents))
        self.assertTrue(all(isinstance(doc, str) for doc in documents_reconstructed))

if __name__ == "__main__":
    STOP_WORDS_DOC = 'how are you'
    # Test with default (no) stop words
    vectorizer = CountVectorizer()
    print("Default stop words:", vectorizer.fit_transform([STOP_WORDS_DOC, STOP_WORDS_DOC]))

    # Test with no stop words
    vectorizer = CountVectorizer(stop_words=[])
    print("No stop words:", vectorizer.fit_transform([STOP_WORDS_DOC, STOP_WORDS_DOC]))

    # Test with custom stop words
    vectorizer = CountVectorizer(stop_words=['are'])
    print("Custom stop words:", vectorizer.fit_transform([STOP_WORDS_DOC, STOP_WORDS_DOC]))

    # Run the unit tests
    unittest.main(argv=[''], exit=False)
