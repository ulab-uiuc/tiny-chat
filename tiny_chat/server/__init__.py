from .config import ConfigManager, EvaluatorConfig, ModelProviderConfig, ServerConfig
from .core import TinyChatServer
from .plugins import EvaluatorPlugin, LLMEvaluatorPlugin, PluginManager, RuleBasedPlugin
from .providers import (
    BaseModelProvider,
    LiteLLMProvider,
    ModelProviderFactory,
    WorkflowProvider,
)

__all__ = [
    'TinyChatServer',
    'ServerConfig',
    'ModelProviderConfig',
    'EvaluatorConfig',
    'ConfigManager',
    'BaseModelProvider',
    'LiteLLMProvider',
    'WorkflowProvider',
    'ModelProviderFactory',
    'EvaluatorPlugin',
    'PluginManager',
    'LLMEvaluatorPlugin',
    'RuleBasedPlugin',
]

# Optional API import - only if FastAPI is available
try:
    from .api import app

    __all__.append('app')
except ImportError:
    # FastAPI not available, app will be None
    app = None
