from abc import ABC, abstractmethod
from typing import Any


class EvaluatorPlugin(ABC):
    """Base class for evaluator plugins"""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    async def evaluate(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Evaluate conversation turn"""
        pass

    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """Return plugin type identifier"""
        pass

    def get_terminal_evaluator(self) -> Any:
        """Return terminal evaluator if this plugin supports it"""
        return None
