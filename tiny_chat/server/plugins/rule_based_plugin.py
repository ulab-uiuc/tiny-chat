import logging
from typing import Any

from tiny_chat.evaluator import RuleBasedTerminatedEvaluator

from .base import EvaluatorPlugin

logger = logging.getLogger(__name__)


class RuleBasedPlugin(EvaluatorPlugin):
    """Rule-based evaluator plugin using TinyChat's existing evaluator"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.max_turn_number = config.get('max_turn_number', 20)
        self.max_stale_turn = config.get('max_stale_turn', 2)

    @property
    def plugin_type(self) -> str:
        return 'rule_based'

    def _create_evaluator(self) -> RuleBasedTerminatedEvaluator:
        """Create the underlying rule-based evaluator"""
        return RuleBasedTerminatedEvaluator(
            max_turn_number=self.max_turn_number, max_stale_turn=self.max_stale_turn
        )
