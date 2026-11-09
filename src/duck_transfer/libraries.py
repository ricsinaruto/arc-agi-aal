"""Clone a function with replaced globals so patched sandbox code never touches the public module."""

from types import FunctionType


def with_globals(function, **overrides):
    """Copy a function's globals so concurrent callers keep their original runner."""
    copied = FunctionType(
        function.__code__,
        {**function.__globals__, **overrides},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied
