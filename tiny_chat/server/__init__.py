from .config import EvaluatorConfig, ModelProviderConfig, ServerConfig
from .core import TinyChatServer
from .providers import BaseModelProvider

__all__ = [
    'TinyChatServer',
    'ServerConfig',
    'ModelProviderConfig',
    'EvaluatorConfig',
    'BaseModelProvider',
]

# Optional API import - only if FastAPI is available
try:
    from .api import app

    __all__.append('app')
except ImportError:
    # FastAPI not available, app will be None
    app = None
