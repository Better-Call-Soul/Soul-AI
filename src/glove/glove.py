import unittest
import torch
from torch import Tensor
from torchnlp.word_to_vector import GloVe

class Glove:
    def __init__(self, TORCHNLP_CACHEDIR='../../models/Glove/pytorch-nlp_data'):
        # The folder where Glove is installed
        self.TORCHNLP_CACHEDIR = TORCHNLP_CACHEDIR
        # define Glove
        self.pretrained_wv = GloVe(cache=self.TORCHNLP_CACHEDIR)
    
    def word2vec(self, word: str) -> torch.Tensor:
        """
        Retrieves the word embedding vector for a given word.

        :param word: The word for which the embedding vector is retrieved.
        :type word: str
        :return: The word embedding vector for the given word.
        :rtype: torch.Tensor
        """
        return self.pretrained_wv[word]

def test():
    class TestGlove(unittest.TestCase):
        
        def setUp(self):
            self.glove_model = Glove()

        def test_glove_existing_word(self):
            word = 'king'
            embedding = self.glove_model.word2vec(word)
            self.assertIsInstance(embedding, Tensor)
            self.assertEqual(embedding.size(0), 300)  # GloVe vectors are of size 300
        
    # Instantiate the test class and run it
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGlove)
    unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
    print('Running unittest...')
    test()
    print('Exit...')
