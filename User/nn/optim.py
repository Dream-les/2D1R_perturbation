import numpy as np
from User.nn.module import Module


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
    _loss = np.ndarray
    _grad = np.ndarray
    _activation_functions = {}
    _loss_functions = {}
    _activation_f = str
    _loss_f = str
    _small_signal = np.ndarray
    _y_pred = np.ndarray

    def __init__(self, loss_f: str = "MSE", af: str = "linear"):
        self._activation_functions = {'linear': lambda x: x,
                                      'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
                                      'tanh': lambda x: np.tanh(x),
                                      'relu': lambda x: np.maximum(0, x),
                                      'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
                                      'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
                                      'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
                                      }#TODO: softmax activation function is facing gradient vanishing problem
        self._activation_f = af
        self._loss_functions = {'MSE': lambda y, y_pred: (y - y_pred) ** 2,
                                'MAE': lambda y, y_pred: np.abs(y - y_pred),
                                'cross-entropy': lambda y, y_pred: -np.mean(
                                    y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)),
                                }  # TODO: cross-entropy loss function is not implemented yet
        self._loss_f = loss_f

    def __call__(self, network_in: Module, y: np.ndarray, lr_in: float = 0.001, ):
        y_pred, loss = self.forward(network_in.get_output, y)
        self._y_pred = y_pred
        self._loss = loss
        self._grad = self.small_signal(network_in.get_output, y).reshape(1, -1) * loss * lr_in
        network_in.start_grad(self._grad)  # TODO: check if this is correct
        network_in.backward(self._grad)
        return y_pred, loss

    def forward(self, x: np.ndarray, y: np.ndarray):
        self._network_in = x * 10
        y_pred = self._activation_functions[self._activation_f](self._network_in)
        return y_pred, self._loss_functions[self._loss_f](y, y_pred, )

    def small_signal(self, x: np.ndarray, y: np.ndarray, dx: float = 1e-3) -> np.ndarray:
        _, y1 = self.forward(x + dx, y)
        _, y2 = self.forward(x - dx, y)
        self._small_signal = (y1 - y2) / (2 * dx)
        return self._small_signal


if __name__ == '__main__':
    network = Module()
