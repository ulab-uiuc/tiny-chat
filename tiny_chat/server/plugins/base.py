from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class EvaluatorPlugin(ABC):
    """Base class for evaluator plugins"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def evaluate(
        self, turn_number: int, messages: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
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
