from typing import Any


class Module:
    """Base class for all neural network modules."""

    module: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """

        :rtype: object
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    # def __getattr__(self, name: str) -> Any:
    #     return getattr(self, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def forward(self, *args, **kwargs) -> Any:
        pass

    def backward(self, *args, **kwargs) -> Any:
        pass

    def step(self, *args, **kwargs) -> Any:
        pass

    def limit_weights(self, *args, **kwargs) -> None:
        pass

    def save_model(self, filename: str) -> None:
        pass

    def load_model(self, filename: str) -> None:
        pass

    def action(self, *args, **kwargs) -> Any:
        pass


