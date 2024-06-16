from User.nn import module
from typing import Union
import User.nn.TDOR as TDOR
import numpy as np
from optim import Function


class NonlinearNetwork(module.Module):
    """
    A Nonlinear_Network is a Module that consists of multiple layers of TDOR.Behaviour.
    """
    __name__ = "Nonlinear_Network"
    layers: list[Union[TDOR.Behaviour, TDOR.Layer2Layer, TDOR.Linear]] = [TDOR.Behaviour(input_dim=1, output_dim=1)]
    _loss_value: np.ndarray = np.array([])
    _network_grad: dict[str, np.ndarray] = {}
    _layer_num: int = 0
    lr: float = 0.01

    def __init__(self, set_layers: list[Union[TDOR.Behaviour, TDOR.Layer2Layer, TDOR.Linear]] = None, ):
        super(NonlinearNetwork, self).__init__()

        if set_layers is None:
            self.layers = [TDOR.Behaviour(28*28, 100), TDOR.Layer2Layer(), TDOR.Behaviour(100, 10), TDOR.Layer2Layer()]
        else:
            self.layers = set_layers

        self._layer_num = len(self.layers)

        self._network_output: np.ndarray = np.array([])

    def add_layer(self, layer: TDOR.Behaviour | TDOR.Layer2Layer | TDOR.Linear):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        self._network_output = x
        return x

    def backward(self, loss_value: np.ndarray, lr: float = 0.01) -> None:#TODO: Check if this is correct
        """
        This function calculates the gradients of the loss function with respect to the parameters of the network.

        :param lr: learning rate
        :param loss_value:
        :return: self._network_grad: A dictionary containing the gradients of the loss function with respect to the
                parameters of the network.
        """
        self.lr = lr
        self._loss_value = loss_value
        current_grad = loss_value * self.lr
        i = self._layer_num
        for layer in reversed(self.layers):
            self._network_grad['layer'+str(i)] = current_grad
            i -= 1
            grad_back = layer.backward(self.current)
            current_grad = grad_back

    def step(self, ) -> None:
        """
        This function updates the parameters of the network using the gradients calculated in the backward function.

        :return: None
        """
        for layer in self.layers:
            layer.step()

    @property
    def get_output(self):
        return self._network_output


class NeuralNetwork(module.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = []

    def add_layer(self, layer: module.Module):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":#TODO: Design a test case for NonlinearNetwork
    nn = NonlinearNetwork()
    loss_fn = Function()
    loss = loss_fn(nn.forward(1), 2)
    print(nn.forward(1))
