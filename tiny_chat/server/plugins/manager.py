import logging
from typing import Any, Dict, List, Optional

from ..config import EvaluatorConfig
from ..providers import BaseModelProvider
from .base import EvaluatorPlugin
from .llm_evaluator_plugin import LLMEvaluatorPlugin
from .rule_based_plugin import RuleBasedPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugins for the TinyChat server"""

    def __init__(self):
        self._evaluators: List[EvaluatorPlugin] = []
        self._plugin_registry = {
            'llm': LLMEvaluatorPlugin,
            'rule_based': RuleBasedPlugin,
        }

    def register_plugin(self, plugin_type: str, plugin_class: type) -> None:
        """Register a new plugin type"""
        self._plugin_registry[plugin_type] = plugin_class
        logger.info(f'Registered plugin type: {plugin_type}')

    def create_evaluator(
        self, config: EvaluatorConfig, model_providers: Dict[str, BaseModelProvider]
    ) -> EvaluatorPlugin | None:
        """Create an evaluator plugin from configuration"""
        plugin_class = self._plugin_registry.get(config.type)

        if plugin_class is None:
            logger.warning(f'Unknown evaluator plugin type: {config.type}')
            return None

        try:
            # Prepare plugin configuration
            plugin_config = config.config.copy()

            # Add model provider if needed
            if config.model and config.model in model_providers:
                plugin_config['model_provider'] = model_providers[config.model]
                plugin_config['model_name'] = config.model

            plugin = plugin_class(plugin_config)
            self._evaluators.append(plugin)

            logger.info(f'Created evaluator plugin: {config.type}')
            return plugin

        except Exception as e:
            logger.error(f'Failed to create evaluator plugin {config.type}: {e}')
            return None

    def get_evaluators(self) -> List[EvaluatorPlugin]:
        """Get all loaded evaluator plugins"""
        return self._evaluators.copy()

    def get_evaluator_by_type(self, plugin_type: str) -> List[EvaluatorPlugin]:
        """Get evaluators by plugin type"""
        return [
            plugin for plugin in self._evaluators if plugin.plugin_type == plugin_type
        ]

    def clear_evaluators(self) -> None:
        """Clear all evaluator plugins"""
        self._evaluators.clear()
        logger.info('Cleared all evaluator plugins')

    def get_available_types(self) -> List[str]:
        """Get list of available plugin types"""
        return list(self._plugin_registry.keys())
