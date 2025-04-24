# Defination for testing
import sys
sys.path.append('F:\\Desktop\\temp\\python\\2D1R_PP\\pythonProject')

from typing import Union
import User.nn.TDOR as TDOR
import numpy as np
from User.nn.optim import Function
from User.nn.base.module import Module
import pickle


class Network(Module):
    """
    A Nonlinear_Network is a Module that consists of multiple layers of TDOR.Behaviour.
    """
    __name__ = "Nonlinear_Network"
    layers: list[Union[TDOR.Behaviour, TDOR.Layer2Layer, TDOR.Linear]] = [TDOR.Behaviour(input_dim=1, output_dim=1)]
    _loss_vec: np.ndarray = np.zeros((1, 10))
    _loss_values: np.ndarray = np.array([])
    _accuracy: float = np.array([])
    _network_grad: dict[str, np.ndarray] = {}
    _layer_num: int = 0
    _y_pred: np.ndarray
    _layer_output: dict[str, np.ndarray] = {}
    lr: float = 0.01
    grad: dict[str, np.ndarray] = {}
    _count: int = 0

    def __init__(self, set_layers: list[Union[TDOR.Behaviour, TDOR.Layer2Layer, TDOR.Linear]] = None, ):
        super(Module, self).__init__()

        if set_layers is None:
            self.layers = [TDOR.Behaviour(28 * 28, 100), TDOR.Layer2Layer(), TDOR.Behaviour(100, 10),
                           TDOR.Layer2Layer()]
        else:
            self.layers = set_layers

        self._layer_num = len(self.layers)

        self._network_output: np.ndarray = np.array([])

    def add_layer(self, layer: TDOR.Behaviour | TDOR.Layer2Layer | TDOR.Linear):
        self.layers.append(layer)

    def forward(self, x_in, y_label, f_object: Function, lr: float = 0.01):
        self.lr = lr
        i = 1
        for layer in self.layers:
            x_in = layer(x_in)
            self._layer_output['layer' + str(i)] = x_in
            i += 1
        self._network_output = x_in
        y_pred, l_value = f_object(self, y_label, self.lr)
        self._y_pred = y_pred
        self._loss_vec = np.append(self._loss_vec, l_value, axis=0)
        if y_label.argmax(axis=1) == self._y_pred.argmax(axis=1):
            self._count += 1
        return y_pred, l_value

    def backward(self, last_grad: np.ndarray, ) -> None:  # TODO: Check if this is correct
        """
        This function calculates the gradients of the loss function with respect to the parameters of the network.

        :param lr: learning rate
        :param last_grad: last layer grad
        :return: self._network_grad: A dictionary containing the gradients of the loss function with respect to the
                parameters of the network.
        """
        current_grad = last_grad
        i = self._layer_num
        for layer in reversed(self.layers):
            self._network_grad['layer' + str(i)] = current_grad
            i -= 1
            grad_back = layer.backward(current_grad)
            current_grad = grad_back

    def step(self, ) -> None:
        """
        This function updates the parameters of the network using the gradients calculated in the backward function.

        :return: None
        """
        for layer in self.layers:
            layer.step()

    def limit_weights(self) -> None:
        """
        This function limits the weights of the network to a certain range.
        :return:
        """
        for layer in self.layers:
            layer.limit_weights()

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            # for layer in self.layers:
            #     pickle.dump(layer, f)

    # def model_eval(self):
    #     self._accuracy = self._count  # / len(self._loss_vec)
    #     self._loss_values = self._loss_vec

    # def load_model(self, filename: str) -> None:
    #     # s = reversed(range(len(self.layers)))
    #     with open(filename, 'rb') as f:
    #         self = pickle.load(f)
    #         # for i in s:
    #         #     self.layers[i].__dict__.update(pickle.load(f).__dict__)

    def start_grad(self, grad: np.ndarray) -> None:
        self._network_grad['loss_grad'] = grad

    def reset_count_loss(self) -> None:
        self._count = 0
        self._loss_vec = np.zeros((1, 10))

    @property
    def get_output(self):
        return self._network_output

    @property
    def get_grad(self) -> dict[str, np.ndarray]:
        return self._network_grad

    @property
    def get_loss(self) -> np.ndarray:
        return self._loss_vec

    @property
    def get_y_pred(self) -> np.ndarray:
        return self._y_pred

    @property
    def get_count(self) -> int:
        return self._count


class NeuralNetwork(Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = []

    def add_layer(self, layer: Module):
        self.layers.append(layer)

    def forward(self, x_in):
        for layer in self.layers:
            x_in = layer(x_in)
        return x_in


if __name__ == "__main__":  # TODO: Design a test case for NonlinearNetwork
    import matplotlib.pyplot as plt
    ############ Nonlinear Layer Test ############
    layer_list = [TDOR.Layer2LayerB(), 
                  TDOR.Behaviour(10, 10), TDOR.Layer2Layer(), ]
    nn = Network(layer_list)
    f_obj = Function("MSE", "sigmoid", )
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    loss_value = []
    for i in range(1000):
        y_p, l = nn.forward(x, y, f_obj, lr=10000)
        print("stop")
        nn.step()
        print('Times:{}'.format(i))
        print(f"y - y_pred: {y - y_p}")
        print(f"label_pred: {y_p.argmax(axis=1)}")
    loos_record = nn.get_loss
    loss_value = np.mean(nn.get_loss, axis=1)[1:]
    plt.plot(loss_value, 'r-o')
    plt.show()
    print(nn.grad)

    ############ Linear Layer Test ############
    '''layer_list = [TDOR.Linear(10, 10), TDOR.Layer2Layer(), 
                  TDOR.Linear(10, 10), TDOR.Layer2Layer(), ]
    nn = Network(layer_list)
    f_obj = Function("MSE", "sigmoid", )
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    loss_value = []
    for i in range(300):
        y_p, l = nn.forward(x, y, f_obj, lr=1e-3)
        print("stop")
        nn.step()
        print('Times:{}'.format(i))
        print(f"y - y_pred: {y - y_p}")
        print(f"label_pred: {y_p.argmax(axis=1)}")
    loos_record = nn.get_loss
    loss_value = np.mean(nn.get_loss, axis=1)[1:]
    plt.plot(loss_value, 'r-o')
    plt.show()
    print(nn.grad)'''
