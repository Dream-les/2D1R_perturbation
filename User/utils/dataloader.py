from typing import Tuple, Any, Dict
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
        self.abspath = 'F:\\Desktop\\temp\\python\\2D1R_PP\\pythonProject\\Dataset\\'
        self.train = dict()
        self.test = dict()
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []

    def __call__(self, name: str = 'optdigits', preprocess: bool = False) -> tuple[dict[Any, Any], dict[Any, Any]]:
        self.load(name)
        if preprocess:
            self.preprocess(name)
        return self.train, self.test

    def load(self, name) -> np.array:
        """
        Loads the dataset from a file.

        Args:
             name (str): The name of the dataset to load.
        Return:
             A numpy array of the dataset.
        """
        train_path = self.abspath + name + '.tra'
        test_path = self.abspath + name + '.tes'
        with open(train_path, 'r') as f:
            for line in f:
                # self.train = np.vstack(self.train, line.strip())
                temp = (line.split(','))
                self.train_data.append(temp[0:-1])
                self.train_labels.append(temp[-1])
                # print(line)
        with open(test_path, 'r') as f:
            for line in f:
                # self.test = np.vstack(self.test, line.strip())
                temp = (line.split(','))
                self.test_data.append(temp[0:-1])
                self.test_labels.append(temp[-1])
                # print(line)

    def preprocess(self, name, one_hot: bool = True) -> None:
        self.train_data = np.array(self.train_data, dtype=float)
        self.test_data = np.array(self.test_data, dtype=float)
        self.train_labels = np.array(self.train_labels, dtype=int).reshape(-1, 1, 1)
        self.test_labels = np.array(self.test_labels, dtype=int).reshape(-1, 1, 1)
        if one_hot:
            temp_train_labels = np.zeros((self.train_labels.shape[0], 1, 10))
            idx = self.train_labels.copy()[:, 0, 0]
            temp_train_labels[np.arange(self.train_labels.shape[0]), 0, idx] = 1
            self.train_labels = temp_train_labels

            temp_test_labels = np.zeros((self.test_labels.shape[0], 1, 10))
            idx = self.test_labels.copy()[:, 0, 0]
            temp_test_labels[np.arange(self.test_labels.shape[0]), 0, idx] = 1
            self.test_labels = temp_test_labels

        self.train_data = self.train_data.reshape(-1, 8, 8)
        self.test_data = self.test_data.reshape(-1, 8, 8)
        self.train_data = self.train_data / 255.0
        self.test_data = self.test_data / 255.0
        self.train["data"] = self.train_data
        self.train["target"] = self.train_labels
        self.test["data"] = self.test_data
        self.test["target"] = self.test_labels

    def __iter__(self):
        pass

    def __next__(self): # TODO:Implement the shuffle()
        '''
        This function is used to iterate over the dataset.
        :return:
        '''
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num = 101
    optd = Datasets()
    train, test = optd('optdigits', preprocess=True)
    plt.imshow(train['data'][num, :, :])
    plt.title(train['target'][num, 0, :].argmax())
    plt.show()
    print(train.shape, test.shape)
