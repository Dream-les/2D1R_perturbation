import numpy as np
from .module import Module
from .TDOR import Behaviour
from .NonlinearNetwork import NonlinearNetwork


def gradient_descent(f, x0, step_size, num_iters, callback=None):
    x = x0
    for i in range(num_iters):
        grad = gradient(f, x)
        x -= step_size * grad
        if callback is not None:
            callback(x)
    return x


def gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def perturbation(f, x0, step_size, num_iters, callback=None):  # TODO: implement perturbation
    x = x0
    for i in range(num_iters):
        grad = gradient(f, x)
        x += step_size * grad
        if callback is not None:
            callback(x)
    return x


class Function:
    def __init__(self, f: str = "MSE"):
        self._functions = {'MSE': lambda y, y_pred: np.mean((y - y_pred) ** 2),
                           'MAE': lambda y, y_pred: np.mean(np.abs(y - y_pred)),
                           'cross-entropy': lambda y, y_pred: -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)),
                           }
        self.f = f

    def __call__(self, network: NonlinearNetwork, y: np.ndarray):
        loss = self.forward(network.get_output(), y)
        network.backward(loss)
        return loss

    def forward(self, network: NonlinearNetwork, y: np.ndarray):
        return self._functions[self.f](network.get_output(), y, )
