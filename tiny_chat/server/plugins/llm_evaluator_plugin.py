import logging
from typing import Any

from tiny_chat.evaluator import EpisodeLLMEvaluator, SotopiaDimensions
from tiny_chat.server.providers import BaseModelProvider

from .base import EvaluatorPlugin

logger = logging.getLogger(__name__)


class LLMEvaluatorPlugin(EvaluatorPlugin):
    """LLM-based evaluator plugin using TinyChat's existing evaluator"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_provider: BaseModelProvider | None = config.get('model_provider')
        self.dimensions = config.get('dimensions', 'sotopia')

    @property
    def plugin_type(self) -> str:
        return 'llm'

    def _create_evaluator(self) -> EpisodeLLMEvaluator[SotopiaDimensions]:
        """Create the underlying LLM evaluator"""
        if self.dimensions == 'sotopia':
            return EpisodeLLMEvaluator[SotopiaDimensions](
                model_provider=self.model_provider
            )
        else:
            # For now, default to SotopiaDimensions
            return EpisodeLLMEvaluator[SotopiaDimensions](
                model_provider=self.model_provider
            )
