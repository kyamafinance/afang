from typing import Callable, List


class FunctionGroup:
    """Defines a function group as a list of functions that can be executed
    sequentially from a single call to the function group."""

    def __init__(self) -> None:
        """Initialize FunctionGroup class."""
        self.funcs: List[Callable] = []

    def add(self, func) -> Callable:
        """Add a function to the function group.

        :param func: function to be added.
        :return: Callable
        """
        self.funcs.append(func)
        return func

    def __call__(self, *args, **kwargs) -> List[Callable]:
        """Sequentially call all functions in the function group.

        :return: List[Callable]
        """
        return [func(*args, **kwargs) for func in self.funcs]
