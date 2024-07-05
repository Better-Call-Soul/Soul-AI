from typing import List
import pickle
import torch


class Helper:
    def __init__(self):
        pass

    def dump_tuple(self, filename: str, data: tuple) -> None:
        '''
        Dump the tuple to a file.
        :param filename: The name of the file to dump the tuple to.
        :type filename: str
        :param data: The tuple to dump.
        :type data: tuple
        '''
        with open(filename, 'wb') as file:
            pickle.dump(data, file)


    def load_tuple(self, filename: str) -> tuple:
        '''
        Load the tuple from the file.
        :param filename: The name of the file to load the tuple from.
        :type filename: str
        :return: The loaded tuple.
        :rtype: tuple
        '''
        with open(filename, 'rb') as file:
            return pickle.load(file)


    def dump_var(self, filename: str, data: List[List[torch.Tensor]]) -> None:
        '''
        Dump the data to a file.
        :param filename: The name of the file to dump the data to.
        :type filename: str
        :param data: The data dump.
        '''
        torch.save(data, filename)


    def load_var(self, filename: str) -> List[List[torch.Tensor]]:
        '''
        Load the data from the file.
        :param filename: The name of the file to load the data from.
        :type filename: str
        :return: The loaded data.
        :rtype: Any
        '''
        return torch.load(filename)
