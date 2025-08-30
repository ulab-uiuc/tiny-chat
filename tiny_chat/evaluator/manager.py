import logging
from typing import TYPE_CHECKING, Any

from .base import BaseEvaluator

if TYPE_CHECKING:
    from tiny_chat.providers import BaseModelProvider

logger = logging.getLogger(__name__)


class EvaluatorConfig:
    def __init__(
        self,
        type: str,
        config: dict[str, Any] | None = None,
        model: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.type = type
        self.config = config or {}
        self.model = model
        self.enabled = enabled


class EvaluatorManager:
    def __init__(self) -> None:
        self._evaluators: list[BaseEvaluator] = []
        self._evaluator_registry: dict[str, type[BaseEvaluator]] = {}
        self._register_builtin_evaluators()

    def _register_builtin_evaluators(self) -> None:
        from .llm_evaluator import LLMEvaluator
        from .rule_based import RuleBasedEvaluator

        self._evaluator_registry['llm'] = LLMEvaluator
        self._evaluator_registry['rule_based'] = RuleBasedEvaluator

    def register_evaluator(
        self, evaluator_type: str, evaluator_class: type[BaseEvaluator]
    ) -> None:
        self._evaluator_registry[evaluator_type] = evaluator_class
        logger.info(f'Registered evaluator type: {evaluator_type}')

    def create_evaluator(
        self,
        config: EvaluatorConfig,
        model_providers: dict[str, 'BaseModelProvider'] | None = None,
    ) -> BaseEvaluator | None:
        evaluator_class = self._evaluator_registry.get(config.type)

        if evaluator_class is None:
            logger.warning(f'Unknown evaluator type: {config.type}')
            return None

        try:
            evaluator_config = config.config.copy()

            if config.model and model_providers and config.model in model_providers:
                evaluator_config['model_provider'] = model_providers[config.model]

            evaluator = evaluator_class(evaluator_config)
            self._evaluators.append(evaluator)

            logger.info(f'Created evaluator: {config.type}')
            return evaluator

        except Exception as e:
            logger.error(f'Failed to create evaluator {config.type}: {e}')
            return None

    def get_evaluators(self) -> list[BaseEvaluator]:
        return self._evaluators.copy()

    def get_evaluators_by_type(self, evaluator_type: str) -> list[BaseEvaluator]:
        return [
            evaluator
            for evaluator in self._evaluators
            if evaluator.evaluator_type == evaluator_type
        ]

    def clear_evaluators(self) -> None:
        self._evaluators.clear()
        logger.info('Cleared all evaluators')

    def get_available_types(self) -> list[str]:
        return list(self._evaluator_registry.keys())
