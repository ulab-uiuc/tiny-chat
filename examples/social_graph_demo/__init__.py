"""
Social Graph Knowledge Propagation Demo

A benchmark for testing knowledge propagation in multi-agent social networks.
"""

from .demo_knowledge_evaluator import DemoKnowledgeEvaluator
from .knowledge_agents import KnowledgeAgent
from .private_conversation_environment import (
    PrivateConversationEnvironment,
    PrivateConversationTinyChatEnvironment,
)

__version__ = '1.0.0'
__all__ = [
    'DemoKnowledgeEvaluator',
    'KnowledgeAgent',
    'PrivateConversationEnvironment',
    'PrivateConversationTinyChatEnvironment',
]
