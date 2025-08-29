from abc import ABC, abstractmethod
from typing import Any

from tiny_chat.evaluator import Evaluator


class EvaluatorPlugin(ABC):
    """Base class for evaluator plugins that wraps TinyChat evaluators"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._evaluator: Evaluator | None = None

    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """Return plugin type identifier"""
        pass

    @property
    def evaluator(self) -> Evaluator:
        """Get the underlying evaluator instance"""
        if self._evaluator is None:
            self._evaluator = self._create_evaluator()
        return self._evaluator

    @abstractmethod
    def _create_evaluator(self) -> Evaluator:
        """Create the underlying evaluator instance"""
        pass

    def _convert_messages(
        self, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, Any]]:
        """Convert messages to the format expected by evaluators"""
        converted_messages = []
        for sender, msg in messages:
            if hasattr(msg, 'to_natural_language'):
                converted_messages.append((sender, msg))
            else:
                from tiny_chat.messages import SimpleMessage

                converted_messages.append((sender, SimpleMessage(message=str(msg))))
        return converted_messages

    async def evaluate(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Evaluate conversation turn using the underlying evaluator"""
        try:
            converted_messages = self._convert_messages(messages)
            return await self.evaluator.__acall__(turn_number, converted_messages)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f'{self.plugin_type} evaluation failed: {e}')
            return []

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Synchronous evaluation"""
        try:
            converted_messages = self._convert_messages(messages)
            return self.evaluator(turn_number, converted_messages)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f'{self.plugin_type} evaluation failed: {e}')
            return []

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Async evaluation"""
        return await self.evaluate(turn_number, messages)

    def get_terminal_evaluator(self) -> Evaluator | None:
        """Return the underlying evaluator for terminal evaluation"""
        return self.evaluator
