from typing import Callable, Protocol


class GenericModel(Protocol):
    fit: Callable
    predict: Callable
