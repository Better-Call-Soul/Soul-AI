from typing import List, Tuple, Callable
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_data(data: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    '''
    Collate function to be used in the DataLoader.
    :param data: The data to collate.
        - inputs (torch.Tensor): The input data.
        - targets (torch.Tensor): The target data.
        - lengths (int): The length of the sequence.
    :type data: list
    :return: A tuple containing the inputs, targets, and masks.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    '''
    # unpack data
    inputs, targets, lengths = zip(*data)
    # convert to lists
    inputs = list(inputs)
    targets = list(targets)
    lengths = list(lengths)

    # pad the targets with -1
    # didn't use zeros as it is a class in the dataset
    targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    # create mask for the loss calculating
    masks = torch.zeros(targets.size(), dtype=torch.bool).to(DEVICE)
    # set mask values to True for valid data
    for i, length in enumerate(lengths):
        masks[i][:length] = 1

    return (inputs, targets, masks)

class EDCDataset(Dataset):
    def __init__(self, inputs: list, targets: list):
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

    def __getitem__(self, index: int) -> tuple:
        '''
        Get an item from the dataset.

        :param index: Index of the item to retrieve.
        :type index: int
        :return: A tuple containing input data, target data, and the length of the target sequence.
        :rtype: tuple
        '''
        selected_inputs = []
        for sentence in self.inputs[index]:
            # the use of stack here to put the vectors of the whole converstation in a sequence
            selected_inputs.append(torch.stack(sentence).to(DEVICE))

        selected_inputs = pack_sequence(selected_inputs, enforce_sorted=False)
        target = torch.LongTensor(self.targets[index]).to(DEVICE)

        return (selected_inputs, target, target.size(0))

    def __len__(self) -> int:
        '''
        Get the length of the dataset.

        :return: The number of samples in the dataset.
        :rtype: int
        '''
        return len(self.inputs)