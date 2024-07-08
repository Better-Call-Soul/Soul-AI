import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    '''
    Improved Model class for Suicidal Detection and Classification.
    '''
    def __init__(self, embedding_size: int, lstm_hidden_size: int, hidden_layer_size: int, cnn_layer_size: int, class_layer: int, dropout_prob: float = 0.5):
        '''
        Initialize the Improved Suicidal Detection and Classification model.
        :param embedding_size: The size of the word embeddings.
        :param lstm_hidden_size: The size of the hidden layer in the LSTM.
        :param hidden_layer_size: The size of the hidden layer in the feedforward neural network.
        :param cnn_layer_size: The size of the CNN layer.
        :param class_layer: The number of classes.
        :param dropout_prob: The probability of dropping neurons during training.
        '''
        super(Model, self).__init__()

        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * lstm_hidden_size, 1)
        self.gmp = nn.AdaptiveMaxPool1d(cnn_layer_size)
        self.fc1 = nn.Linear(2 * lstm_hidden_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, class_layer)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the improved model.
        :param x: The input to the model.
        :return: The output of the model.
        '''
        lstm_out, _ = self.lstm(x)

        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        attn_out = torch.sum(attn_weights * lstm_out, dim=1)

        x = self.gmp(attn_out.unsqueeze(2)).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x