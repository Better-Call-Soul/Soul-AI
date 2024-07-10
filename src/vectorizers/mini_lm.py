
from sentence_transformers import SentenceTransformer

# MiniLM vectorizer
# This class is a wrapper for the MiniLM model from the Sentence Transformers library
class MiniLM:
    # initialize the model
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # fit the vectorizer to the data
    def encode(self, statements):
        # output the embeddings of the statements
        # the feature vectors are normalized to have a length of 384-dimensional dense vector space
        return self.model.encode(statements)
    
