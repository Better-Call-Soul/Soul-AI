import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    '''
    Model class for the Suicidal Detection and Classification model.
    '''

    def __init__(self, embedding_size: int, lstm_hidden_size: int, hidden_layer_size: int, cnn_layer_size: int, class_layer: int):
        '''
        Initialize the Suicidal Detection and Classification model.
        :param embedding_size: The size of the word embeddings.
        :type embedding_size: int
        :param lstm_hidden_size: The size of the hidden layer in the LSTM.
        :type lstm_hidden_size: int
        :param hidden_layer_size: The size of the hidden layer in the feedforward neural network.
        :type hidden_layer_size: int
        :param cnn_layer_size: The size of the CNN layer.
        :type cnn_layer_size: int
        :param class_layer: The number of classes.
        :type class_layer: int
        '''
        super(Model, self).__init__()

        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True)
        self.gmp = nn.AdaptiveMaxPool1d(cnn_layer_size)
        self.fc1 = nn.Linear(lstm_hidden_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, class_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        :param x: The input to the model.
        :type x: torch.Tensor
        :return: The output of the model.
        :rtype: torch.Tensor
        '''
        x, _ = self.lstm(x)
        x = self.gmp(x.transpose(1, 2)).squeeze(2)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x