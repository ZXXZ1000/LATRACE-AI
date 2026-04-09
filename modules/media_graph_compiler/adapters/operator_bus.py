from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional


OperatorFn = Callable[[Mapping[str, Any]], Dict[str, Any]]


class OperatorBus:
    """Registry for pluggable stages such as visual, speaker, and semantic operators."""

    def __init__(self) -> None:
        self._registry: Dict[str, OperatorFn] = {}

    def register(self, name: str, func: OperatorFn) -> None:
        self._registry[str(name)] = func

    def get(self, name: str) -> Optional[OperatorFn]:
        return self._registry.get(str(name))

    def clear(self) -> None:
        self._registry.clear()


default_operator_bus = OperatorBus()


def register_operator(name: str, func: OperatorFn) -> None:
    default_operator_bus.register(name, func)


def get_operator(name: str) -> Optional[OperatorFn]:
    return default_operator_bus.get(name)


def clear_operators() -> None:
    default_operator_bus.clear()
