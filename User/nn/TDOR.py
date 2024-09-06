import pickle

import numpy as np
from User.nn.module import Module
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
    _get_grad: float
    _layer_output: np.ndarray
    _deliver_grad: np.ndarray
    _dy_dx: np.ndarray
    _dy_dw: np.ndarray
    x: np.ndarray
    fx: np.ndarray
    temp: np.ndarray

    def __init__(self, input_dim=4, output_dim=4, delta: float = 1e-10, ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.weights = np.random.rand(input_dim, output_dim) * 0.01 - 0.05  # alpha: range(0.04,0.09)
        # self.delta_x = np.zeros((input_dim, )) + delta
        self.delta_x = delta
        self._type = "Behaviour"

    def action(self, x_in: np.ndarray, d_w=0.0) -> np.ndarray:
        """
        pass the input through the weights.

        :param d_w: 
        :param: beta = -3.2944 * alpha + 42.59
        :param: alpha -> weight, range(0.04,0.09)
        :param x_in: np.ndarray((1, input_dim)), a row vector of input,(-0.1,0.1)
        :return: 1E-6 * alpha * ((torch.exp(beta * x) - 1))
        """
        beta = -3.2944 * (self.weights + d_w) + 42.59
        temp = np.multiply(x_in.transpose(), beta)
        self.fx = 1E5 * 1E-2 * 1E-6 * (self.weights + d_w) * (np.exp(temp) - np.exp(-temp))  # TODO: Negative gradient?
        return self.fx

    def forward(self, x_in: np.ndarray, ) -> np.ndarray:
        # self.delta_x = np.zeros((x.shape[1], )) + 0.01
        self.x = x_in.copy()
        self.small_signal(self.x)
        self._layer_output = self.action(x_in)
        self._layer_output = self._layer_output.sum(axis=0).reshape(1, self._output_dim)
        return self._layer_output

    def small_signal(self, x: np.ndarray, ):
        """
        pass the input through the weights to get the small signal approximation of the derivative.

        :param x:
        :return: None
        """
        # self._layer_output = self.forward(x)
        d_yx = self.action(x + self.delta_x) - self.action(x - self.delta_x)
        self._dy_dx = d_yx / (2 * self.delta_x)
        d_yw = self.action(x, self.delta_x) - self.action(x, -self.delta_x)
        self._dy_dw = d_yw / (2 * self.delta_x)  # TODO: Negative gradient?

    def backward(self, last_grad: float) -> np.ndarray:
        """
        Get the gradient from the derivative of the loss w.r.t. the output of the layer
        Deliver the gradient to the previous layer.

        :param last_grad:
        :return: None
        """
        self._get_grad = last_grad
        # self._deliver_grad = self._dy_dx * loss_value
        self._deliver_grad = last_grad @ self.weights.transpose()  # TODO: Check special backward
        return self._deliver_grad

    def step(self, ) -> None:
        """
        update the weights according to the small signal approximation of the derivative.

        :param any:
        :return: None
        """
        # self.weights += self._dy_dx * self._get_grad # y-y_pred
        one = np.ones((1, self._input_dim))
        self.temp = (one.transpose() @ self._get_grad) * self._dy_dw  # * self.x.transpose()
        self.weights -= self.temp  # TODO: Check special backward

    def limit_weights(self, ) -> None:
        # self.weights = np.clip(self.weights, -1, 1)
        val = np.max(self.weights) - np.min(self.weights)
        if val == 0:
            raise ValueError("The weights are all the same, cannot limit them")
        self.weights = (self.weights / val) * 0.1
        # self.weights = (self.weights / val) * 0.05 + 0.04  # TODO: Revise the range of alpha

    @property
    def type(self):
        return self._type

    @property
    def dy_dw(self):
        return self._dy_dw

    @property
    def get_output(self):
        return self._layer_output

    @property
    def deliver_grad(self):
        return self._deliver_grad

    @property
    def get_grad(self):
        return self._get_grad

    @validate_instance(float, Any)
    def set_loss_value(self, loss_value: float) -> float:
        self._get_grad = loss_value
        return self._get_grad


class Layer2Layer(Module):
    """
    This is a template class for the layer-to-layer connection of 2D1R.
    """
    _layer_output = np.ndarray

    def __init__(self, input_dim=1, output_dim=1, ):
        super().__init__()
        self.weights = np.random.randn(input_dim, output_dim)

    def scaling(self, x_in: np.ndarray, ) -> np.ndarray:  # TODO: change to small_signal
        """
        pass the input through the weights.

        :param x_in:
        :return:
        """
        # 0.1 * (np.exp(x_in) / np.sum(np.exp(x_in)))
        val = np.max(x_in)-np.min(x_in)
        self._x_in = x_in.copy()
        self._val = val
        if val == 0:
            val = 1
        return 0.2 * ((x_in - np.min(x_in)) / val) - 0.1

    def forward(self, x_in: np.ndarray) -> np.ndarray:
        self._layer_output = self.scaling(x_in)
        return self._layer_output

    def backward(self, last_grad: np.ndarray) -> np.ndarray:
        return last_grad

    def step(self, ) -> None:
        pass

    def limit_weights(self, ) -> None:
        # self.weights = np.clip(self.weights, -1, 1)
        val = np.max(self.weights) - np.min(self.weights)
        if val == 0:
            val = 1
        self.weights = (self.weights / val) * 0.1


class Linear(Module):
    _dy_dx: np.ndarray
    _dy_dw: np.ndarray
    _get_grad: float
    _layer_output: np.ndarray
    _deliver_grad: np.ndarray
    _x_in: np.ndarray
    delta: float = 1e-6
    delta_w: np.ndarray

    def __init__(self, input_dim=1, output_dim=1, ):
        super().__init__()
        self.weights = np.random.rand(input_dim, output_dim)

    def forward(self, x_in: np.ndarray, delta_w: float = 0.0) -> np.ndarray:
        self._layer_output = self.action(x_in, delta_w).reshape(1, -1)
        self.small_signal(x_in)
        return self._layer_output

    def action(self, x_in: np.ndarray, delta_w: float = 0.0) -> np.ndarray:
        self._x_in = x_in.copy()
        return np.dot(x_in, self.weights + delta_w)

    def backward(self, last_grad: float) -> np.ndarray:
        self._get_grad = last_grad
        self._deliver_grad = self._get_grad @ self.weights.transpose()
        return self._deliver_grad

    def step(self, ) -> None:
        """
        update the weights according to the small signal approximation of the derivative.

        :param lr_in:
        :param any:
        :return: None
        """
        self.delta_w = self._x_in.transpose() @ self._get_grad  # * self.x.transpose()
        self.weights -= self.delta_w  # TODO: Check special backward

    def small_signal(self, x_in: np.ndarray, ):  # TODO: change to small_signal, deal with the sum of every column
        """
        pass the input through the weights to get the small signal approximation of the derivative.

        :param x_in:
        :return: None
        """
        # self._layer_output = self.forward(x)
        d_yx = self.action(x_in + self.delta) - self.action(x_in - self.delta)
        self._dy_dx = d_yx / (2 * self.delta)
        d_yw = self.action(x_in, self.delta) - self.action(x_in, -self.delta)
        self._dy_dw = d_yw / (2 * self.delta)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    b = Behaviour(19, 10, 0.01)
    init_w = (np.random.randn(19, 10) * 0.05) + 0.04
    b.weights = init_w.copy()
    b.delta_x = 0.01
    x = np.arange(-0.09, 0.1, 0.01).reshape(1, 19)
    loss = np.arange(-0.05, 0.05, 0.01).reshape(1, 10) * 0.01
    y = b.forward(x)
    # y = b.action(x)
    dy = b.dy_dw
    dil_w = b.backward(loss)
    b.step()
    a = b.deliver_grad
    c = b.weights - init_w
    # c[9, :] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dy)
    try:
        # ax2.semilogy(abs(c[0, :]), 'r.')
        ax2.imshow(c)
    except ValueError as e:
        print("Check the shape of dy_dx")
    plt.show()
    d = dy * x.transpose()
    filepath= 'F:\\Desktop\\temp\\python\\2D1R_PP\\pythonProject\\model\\weights.pkl'
    b.save_model(filepath)
    b.load_model(filepath)
    print(y)
