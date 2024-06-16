import numpy as np
from .module import Module
from typing import Any


def validate_instance(instance, value):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not isinstance(value, instance):
                raise ValueError(f"Value must be an instance of {instance}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class Behaviour(Module):
    """
    This is a template class for the behaviour of 2D1R.
    """
    _dy_dx: np.ndarray
    _get_grad: float
    _layer_output: np.ndarray
    _deliver_grad: np.ndarray

    def __init__(self, input_dim=2, output_dim=2, delta: float = 0.01, ):
        super().__init__()
        self.weights = np.random.randn(input_dim, output_dim)
        self.delta_x = np.zeros((input_dim, )) + delta
        self._type = "Behaviour"

    def action(self, x: np.ndarray, ) -> np.ndarray:
        """
        pass the input through the weights.

        :param: beta = -3.2944 * alpha + 42.59
        :param: alpha -> weight,
        :param x: np.ndarray((1, input_dim)), a row vector of input
        :return: 1E-6 * alpha * ((torch.exp(beta * x) - 1))
        """
        beta = -3.2944 * self.weights + 42.59
        temp = np.matmul(x.transpose(), beta)
        return 1E5 * 1E-2 * 1E-6 * self.weights * (np.exp(temp) - np.exp(-temp))

    def forward(self, x: np.ndarray, ) -> np.ndarray:  #TODO: Check the logic
        self.small_signal(x)
        return self.action(x)

    def small_signal(self, x: np.ndarray, ):  #TODO: change to small_signal, deal with the sum of every column
        """
        pass the input through the weights to get the small signal approximation of the derivative.

        :param x:
        :return: None
        """
        self._layer_output = self.forward(x)
        d_y = self.action(x+self.delta_x) - self.action(x-self.delta_x)
        self._dy_dx = d_y / (2*self.delta_x)

    def backward(self, loss_value: float) -> np.ndarray:#TODO: Check if the loss_value is correct
        """
        Get the gradient from the derivative of the loss w.r.t. the output of the layer
        Deliver the gradient to the previous layer.

        :param loss_value:
        :return: None
        """
        self._get_grad = loss_value
        self._deliver_grad = self._dy_dx * loss_value
        return self._deliver_grad

    def step(self, ) -> None:
        """
        update the weights according to the small signal approximation of the derivative.

        :param any:
        :return: None
        """
        self.weights -= self._dy_dx * self._get_grad

    @property
    def type(self):
        return self._type

    @property
    def dy_dx(self):
        return self._dy_dx

    @property
    def get_output(self):
        return self._layer_output

    @validate_instance(float, Any)
    def set_loss_value(self, loss_value: float) -> float:
        self._get_grad = loss_value
        return self._get_grad


class Layer2Layer(Module):
    """
    This is a template class for the layer-to-layer connection of 2D1R.
    """

    def __init__(self, input_dim=1, output_dim=1, ):
        super().__init__()
        self.weights = np.random.randn(input_dim, output_dim)

    def scaling(self, x: np.ndarray, ) -> np.ndarray:  #TODO: change to small_signal
        """
        pass the input through the weights.

        :param x:
        :return:
        """
        pass


class Linear(Module):

    def __init__(self, input_dim=1, output_dim=1,):
        super().__init__()
        self.weights = np.random.randn(input_dim, output_dim)

    def forward(self, x: np.ndarray, ) -> np.ndarray:
        return np.dot(x, self.weights)


if __name__ == '__main__':
    pass
