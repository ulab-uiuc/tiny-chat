import logging
from typing import Any, Dict, List, Tuple

from ...evaluator import RuleBasedTerminatedEvaluator
from .base import EvaluatorPlugin

logger = logging.getLogger(__name__)


class RuleBasedPlugin(EvaluatorPlugin):
    """Rule-based evaluator plugin using TinyChat's existing evaluator"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.max_turn_number = config.get("max_turn_number", 20)
        self.max_stale_turn = config.get("max_stale_turn", 2)

        # Create the underlying evaluator
        self.evaluator = RuleBasedTerminatedEvaluator(
            max_turn_number=self.max_turn_number, max_stale_turn=self.max_stale_turn
        )

    @property
    def plugin_type(self) -> str:
        return "rule_based"

    async def evaluate(
        self, turn_number: int, messages: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        """Evaluate conversation turn using rules"""
        try:
            converted_messages = []
            for sender, msg in messages:
                if hasattr(msg, "to_natural_language"):
                    converted_messages.append((sender, msg))
                else:
                    from ...messages import SimpleMessage

                    converted_messages.append((sender, SimpleMessage(message=str(msg))))

            result = await self.evaluator.__acall__(
                turn_number=turn_number, messages=converted_messages
            )

            logger.debug(f"Rule-based evaluator returned {len(result)} results")
            return result

        except Exception as e:
            logger.error(f"Rule-based evaluation failed: {e}")
            return []
