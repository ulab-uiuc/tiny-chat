from .core import TinyChatServer

__all__ = [
    'TinyChatServer',
]

try:
    from .api import app

    __all__.append('app')
except ImportError:
    app = None
