import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    '''
    Model class for the Emotion Detection and Classification model.
    '''
    def __init__(self, word_embedding_size: int, lstm_hidden_size: int, hidden_layer_size: int, output_classes: int = 7):
        '''
        Initialize the Model.
        :param word_embedding_size: The size of the word embeddings.
        :param lstm_hidden_size: The size of the LSTM hidden layer.
        :param hidden_layer_size: The size of the hidden layer.
        :param output_classes: The size of the output layer.
        '''
        super().__init__()

        # Feature extraction module
        self.feature_extractor = nn.LSTM(word_embedding_size, lstm_hidden_size, batch_first=True, bidirectional=True)

        # Emotion detection module
        self.global_state_gru = nn.GRUCell(2 * lstm_hidden_size, 2 * lstm_hidden_size)
        self.global_state_weight = nn.Parameter(torch.randn(2 * lstm_hidden_size, 2 * lstm_hidden_size))
        self.party_state_gru = nn.GRUCell(2 * lstm_hidden_size, 2 * lstm_hidden_size)
        self.emotion_state_gru = nn.GRUCell(2 * lstm_hidden_size, hidden_layer_size)

        # Classification module
        self.classifier = nn.Linear(hidden_layer_size, output_classes)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Extract features from the input data using a bidirectional LSTM.
        :param inputs: The input data.
        :return: The output of the feature extraction module.
        '''
        outputs = []
        for input in inputs:
            output, _ = self.feature_extractor(input)
            output, lengths = pad_packed_sequence(output, batch_first=True)
            output = output.view(output.size(0), output.size(1), 2, -1)
            output = torch.stack([torch.cat([output[i, length-1, 0], output[i, 0, 1]]) for i, length in enumerate(lengths)])
            outputs.append(output)
        outputs = pad_sequence(outputs, batch_first=True)
        return outputs

    def detect_emotions(self, utterance_inputs: torch.Tensor) -> torch.Tensor:
        '''
        Detect emotions from the utterance inputs using GRU cells and attention mechanism.
        :param utterance_inputs: The input data.
        :return: The output of the emotion detection module.
        '''
        utterance_inputs = utterance_inputs.transpose(0, 1)
        global_state = [torch.zeros(utterance_inputs.size(1), 2 * self.feature_extractor.hidden_size).to(DEVICE)]
        party_state = [torch.zeros(utterance_inputs.size(1), 2 * self.feature_extractor.hidden_size).to(DEVICE) for _ in range(2)]
        emotion_state = torch.zeros(utterance_inputs.size(1), self.classifier.in_features).to(DEVICE)

        emotion_outputs = []
        for i in range(utterance_inputs.size(0)):
            current_utterance = utterance_inputs[i]
            current_party = i % 2

            global_state.append(self.global_state_gru(current_utterance + party_state[current_party], global_state[i]))
            global_state_history = torch.stack(global_state[:i+1])
            attention_score = (current_utterance.unsqueeze(1) @ self.global_state_weight @ global_state_history.unsqueeze(-1)).squeeze(-1)
            attention_weights = F.softmax(attention_score, dim=0)
            context_vector = torch.sum(global_state_history * attention_weights, dim=0)

            party_state[current_party] = self.party_state_gru(current_utterance + context_vector, party_state[current_party])
            emotion_state = self.emotion_state_gru(party_state[current_party], emotion_state)
            emotion_outputs.append(emotion_state)

        emotion_outputs = torch.stack(emotion_outputs).transpose(0, 1)
        return emotion_outputs

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the Model.
        :param input_data: The input data.
        :return: The output of the Model.
        '''
        feature_output = self.extract_features(input_data)
        emotion_output = self.detect_emotions(feature_output)
        classification_output = self.classifier(emotion_output)
        return classification_output
