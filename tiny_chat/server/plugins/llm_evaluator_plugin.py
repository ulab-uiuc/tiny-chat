import logging
from typing import Any

from ...evaluator import EpisodeLLMEvaluator, SotopiaDimensions
from ..providers import BaseModelProvider
from .base import EvaluatorPlugin

logger = logging.getLogger(__name__)


class LLMEvaluatorPlugin(EvaluatorPlugin):
    """LLM-based evaluator plugin using TinyChat's existing evaluator"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        self.model_provider: BaseModelProvider | None = config.get('model_provider')
        self.dimensions = config.get('dimensions', 'sotopia')

        if self.dimensions == 'sotopia':
            self.evaluator = EpisodeLLMEvaluator[SotopiaDimensions](
                model_provider=self.model_provider
            )
        else:
            self.evaluator = EpisodeLLMEvaluator[SotopiaDimensions](
                model_provider=self.model_provider
            )

    @property
    def plugin_type(self) -> str:
        return 'llm'

    async def evaluate(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Evaluate conversation turn using LLM"""
        try:
            converted_messages = []
            for sender, msg in messages:
                if hasattr(msg, 'to_natural_language'):
                    converted_messages.append((sender, msg))
                else:
                    from ...messages import SimpleMessage

                    converted_messages.append((sender, SimpleMessage(message=str(msg))))

            result = await self.evaluator.__acall__(
                turn_number=turn_number, messages=converted_messages
            )

            logger.debug(f'LLM evaluator returned {len(result)} results')
            return result

        except Exception as e:
            logger.error(f'LLM evaluation failed: {e}')
            return []

    def get_terminal_evaluator(self) -> EpisodeLLMEvaluator[SotopiaDimensions]:
        """Return the underlying evaluator for terminal evaluation"""
        return self.evaluator

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        try:
            converted_messages = self._convert_messages(messages)

            result = self._evaluate_sync(turn_number, converted_messages)

            logger.debug(f'LLM evaluator returned {len(result)} results')
            return result
        except Exception as e:
            logger.error(f'LLM evaluation failed: {e}')
            return []

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Any]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return await self.evaluate(turn_number, messages)
