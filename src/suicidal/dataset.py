from typing import List, Tuple, Callable
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Collate function to be used with DataLoader.
    :param batch: The batch to collate.
    :type batch: List[Tuple[torch.Tensor, torch.Tensor]]
    :return: The padded inputs and targets.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    '''
    # Unzip the batch
    inputs, targets = zip(*batch)

    # Convert inputs to tensors and pad them
    inputs_padded = pad_sequence([torch.stack(seq)
                                 for seq in inputs], batch_first=True)

    # Convert targets to tensor
    targets = torch.tensor(targets)

    return inputs_padded, targets

class SUIDataset(Dataset):
    def __init__(self, inputs: list, targets: list, word2vec: Callable[[str], torch.Tensor]):
        '''
        Initialize the EDCDataset.

        :param inputs: List of conversations, where each conversation is a list of sentences,
                       and each sentence is a list of words.
        :type inputs: list
        :param targets: List of labels for each conversation.
        :type targets: list
        '''
        self.inputs = inputs
        self.targets = targets
        self.word2vec = word2vec

    def __getitem__(self, index: int) -> tuple:
        '''
        Get the item at the specified index.

        :param index: The index of the item to get.
        :type index: int
        :return: The conversation and its label at the specified index.
        :rtype: tuple
        '''
        return [self.word2vec(word) for word in self.inputs[index]], self.targets[index]

    def __len__(self) -> int:
        '''
        Get the length of the dataset.

        :return: The number of samples in the dataset.
        :rtype: int
        '''
        return len(self.inputs)
