import logging
from typing import Any, Dict, List, Optional, Tuple

from ...evaluator import EpisodeLLMEvaluator, TinyChatDimensions
from ...messages import Message
from ..providers import BaseModelProvider
from .base import EvaluatorPlugin

logger = logging.getLogger(__name__)


class LLMEvaluatorPlugin(EvaluatorPlugin):
    """LLM-based evaluator plugin using TinyChat's existing evaluator"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_provider: Optional[BaseModelProvider] = config.get('model_provider')
        self.model_name: str = config.get('model_name', 'gpt-4o-mini')
        self.dimensions = config.get('dimensions', 'sotopia')

        # Create the underlying evaluator
        if self.dimensions == 'sotopia':
            self.evaluator = EpisodeLLMEvaluator[TinyChatDimensions](
                model_name=self.model_name
            )
        else:
            # Default to TinyChatDimensions for now
            self.evaluator = EpisodeLLMEvaluator[TinyChatDimensions](
                model_name=self.model_name
            )

        # Check if we have a valid evaluator
        if not self.evaluator:
            logger.warning(f'Failed to create LLM evaluator for {self.model_name}')
            self.evaluator = None

    @property
    def plugin_type(self) -> str:
        return 'llm'

    async def evaluate(
        self, turn_number: int, messages: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        """Evaluate conversation turn using LLM"""
        try:
            # Convert messages to the format expected by the evaluator
            converted_messages = []
            for sender, msg in messages:
                if hasattr(msg, 'to_natural_language'):
                    converted_messages.append((sender, msg))
                else:
                    # Wrap in a simple message if needed
                    from ...messages import SimpleMessage

                    converted_messages.append((sender, SimpleMessage(message=str(msg))))

            # Call the underlying evaluator
            result = await self.evaluator.__acall__(
                turn_number=turn_number, messages=converted_messages
            )

            logger.debug(f'LLM evaluator returned {len(result)} results')
            return result

        except Exception as e:
            logger.error(f'LLM evaluation failed: {e}')
            return []

    def get_terminal_evaluator(self) -> Any:
        """Return the underlying evaluator for terminal evaluation"""
        if self.evaluator is None:
            logger.warning('LLM evaluator not available, returning None')
            return None
        return self.evaluator
