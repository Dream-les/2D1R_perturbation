from typing import Tuple, Any
import numpy as np
from numpy import ndarray, dtype


class Dataloader:
    """
    A class that implements a DataLoader for loading data from a file.
    Iteration of datasets yields batches of data.

    :param any: The parameters of dataset to load.
    :return: A DataLoader object.
    """
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


class Datasets:
    """
    A class that implements a Datasets class for loading data from a file.
    """
    def __init__(self) -> None:
        self.train = []
        self.test = []

    def __call__(self, name: str = 'optdigits', preprocess: bool = False) -> tuple[ndarray[Any, dtype[int]], ndarray[Any, dtype[int]]]:
        self.load(name)
        if preprocess:
            self.preprocess(name)
        return np.array(self.train, dtype=int), np.array(self.test, dtype=int)

    def load(self, name) -> np.array:
        """
        Loads the dataset from a file.

        Args:
             name (str): The name of the dataset to load.
        Return:
             A numpy array of the dataset.
        """
        train_path = './data/' + name + '.tra'
        test_path = './data/' + name + '.tes'
        with open(train_path, 'r') as f:
            for line in f:
                self.train.append(line.strip())
                print(line)
        with open(test_path, 'r') as f:
            for line in f:
                self.test.append(line.strip())
                print(line)

    def preprocess(self, name) -> None:
        pass


if __name__ == '__main__':
    pass
