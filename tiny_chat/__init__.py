"""
Tiny Chat - A simplified multi-agent chat system inspired by Sotopia
"""

__version__ = '0.0.1'

from .agents import LLMAgent
from .environment import ChatEnvironment
from .evaluators import (
    EpisodeLLMEvaluator,
    EvaluationDimension,
    EvaluationForMultipleAgents,
    EvaluationForTwoAgents,
    Evaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
    unweighted_aggregate_evaluate,
)
from .generator import MessageGenerator
from .messages import AgentAction, ChatBackground, Message, Observation, SimpleMessage
from .profile import AgentProfile, RelationshipProfile, RelationshipType
from .server import ChatServer

__all__ = [
    'LLMAgent',
    'Message',
    'SimpleMessage',
    'Observation',
    'AgentAction',
    'ChatBackground',
    'ChatEnvironment',
    'MessageGenerator',
    'AgentProfile',
    'RelationshipProfile',
    'RelationshipType',
    'Evaluator',
    'RuleBasedTerminatedEvaluator',
    'EpisodeLLMEvaluator',
    'SotopiaDimensions',
    'EvaluationDimension',
    'EvaluationForTwoAgents',
    'EvaluationForMultipleAgents',
    'unweighted_aggregate_evaluate',
    'ChatServer',
]
