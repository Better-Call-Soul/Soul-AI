import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtraction(nn.Module):
    '''
    FeatureExtraction module to extract features from the input data.
    '''
    def __init__(self, embedding_size: int, hidden_layer_size: int):
        '''
        Initialize the FeatureExtraction module.
        :param embedding_size: The size of the word embeddings.
        :type embedding_size: int
        :param hidden_layer_size: The size of the hidden layer.
        :type hidden_layer_size: int
        '''
        super().__init__()
        self.bilstm = nn.LSTM(embedding_size, hidden_layer_size,
                              batch_first=True, bidirectional=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the FeatureExtraction module.
        :param inputs: The input data.
        :type inputs: torch.Tensor
        :return: The output of the FeatureExtraction module.
        :rtype: torch.Tensor
        '''
        outputs = []
        for input in inputs:
            # LSTM output: (N, max(L), D) -> (N, max(L), 2*H)
            output, _ = self.bilstm(input)
            # Unpack the PackedSequence to get the output and lengths: (N, max(L), 2*H)
            output, lengths = pad_packed_sequence(
                output, batch_first=True)
            # Reshape to separate bidirectional outputs: (N, max(L), 2, H)
            output = output.view(output.size(0), output.size(1), 2, -1)
            # Extract last forward and first backward outputs: (N, max(L), 2*H) -> (N, 2*H)
            output = torch.stack([torch.cat([output[i, length-1, 0], output[i, 0, 1]])
                                  for i, length in enumerate(lengths)])
            outputs.append(output)
        # Pad the sequences: (B, max(N), U)
        outputs = pad_sequence(outputs, batch_first=True)
        return outputs
    

class EmotionDetection(nn.Module):
    '''
    Emotion Detection model.
    '''
    def __init__(self, utterance_embedding_size: int, hidden_layer_size: int):
        '''
        Initialize the EmotionDetection module.
        :param utterance_embedding_size: The size of the utterance embeddings.
        :type utterance_embedding_size: int
        :param hidden_layer_size: The size of the hidden layer.
        :type hidden_layer_size: int
        '''
        super().__init__()
        self.utterance_embedding_size = utterance_embedding_size
        self.hidden_layer_size = hidden_layer_size
        # Create the layers
        self.global_state_gru = nn.GRUCell(self.utterance_embedding_size, self.utterance_embedding_size)
        self.global_state_weight = nn.Parameter(torch.randn(self.utterance_embedding_size, self.utterance_embedding_size))
        self.party_state_gru = nn.GRUCell(self.utterance_embedding_size, self.utterance_embedding_size)
        self.emotion_state_gru = nn.GRUCell(self.utterance_embedding_size, self.hidden_layer_size)

    def forward(self, utterance_inputs: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the EmotionDetection module.
        :param inputs: The input data.
        :type inputs: torch.Tensor
        :return: The output of the EmotionDetection module.
        :rtype: torch.Tensor
        '''
        # Transpose inputs to shape (N, B, U)
        utterance_inputs = utterance_inputs.transpose(0, 1)

        # Initialize states
        global_state = [torch.zeros(utterance_inputs.size(1), self.utterance_embedding_size).to(DEVICE)]
        party_state = [torch.zeros(utterance_inputs.size(1), self.utterance_embedding_size).to(DEVICE) for _ in range(2)]
        emotion_state = torch.zeros(utterance_inputs.size(1), self.hidden_layer_size).to(DEVICE)

        emotion_outputs = []
        for i in range(utterance_inputs.size(0)):
            current_utterance = utterance_inputs[i]  # (B, U)
            current_party = i % 2

            # Update global state
            global_state.append(self.global_state_gru(current_utterance + party_state[current_party], global_state[i]))

            # Calculate attention scores
            global_state_history = torch.stack(global_state[:i+1])  # (1+n, B, U)
            attention_score = (current_utterance.unsqueeze(1) @ self.global_state_weight @ global_state_history.unsqueeze(-1)).squeeze(-1)
            attention_weights = F.softmax(attention_score, dim=0)  # (1+n, B, 1)

            # Calculate context vector
            context_vector = torch.sum(global_state_history * attention_weights, dim=0)  # (B, U)

            # Update party and emotion states
            party_state[current_party] = self.party_state_gru(current_utterance + context_vector, party_state[current_party])
            emotion_state = self.emotion_state_gru(party_state[current_party], emotion_state)

            emotion_outputs.append(emotion_state)

        # Transpose outputs to shape (B, N, H)
        emotion_outputs = torch.stack(emotion_outputs).transpose(0, 1)

        return emotion_outputs
    

class Model(nn.Module):
    '''
    Model class for the Emotion Detection and Classification model.
    '''
    def __init__(self, word_embedding_size: int, lstm_hidden_size, hidden_layer_size: int, output_classes: int = 7):
        '''
        Initialize the Model.
        :param word_embedding_size: The size of the word embeddings.
        :type word_embedding_size: int
        :param lstm_hidden_size: The size of the LSTM hidden layer.
        :type lstm_hidden_size: int
        :param hidden_layer_size: The size of the hidden layer.
        :type hidden_layer_size: int
        :param output_classes: The size of the output layer.
        :type output_classes: int
        '''
        super().__init__()

        self.feature_extractor = FeatureExtraction(word_embedding_size, lstm_hidden_size)
        self.emotion_detector = EmotionDetection(2 * lstm_hidden_size, hidden_layer_size)
        self.classifier = nn.Linear(hidden_layer_size, output_classes)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the Model.
        :param inputs: The input data.
        :type inputs: torch.Tensor
        :return: The output of the Model.
        :rtype: torch.Tensor
        '''
        # Extract features from input data
        feature_output = self.feature_extractor(input_data)  # (B, pack(N, L, D)) -> (B, N, U)

        # Detect emotions from feature output
        emotion_output = self.emotion_detector(feature_output)  # (B, N, U) -> (B, N, H)

        # Classify emotions
        classification_output = self.classifier(emotion_output)  # (B, N, H) -> (B, N, C)

        # Use CrossEntropyLoss for classification
        return classification_output