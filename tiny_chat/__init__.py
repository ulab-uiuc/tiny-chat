"""
Tiny Chat - A simplified multi-agent chat system inspired by Sotopia
"""

__version__ = '0.0.1'

from .agents import LLMAgent
from .envs import TinyChatEnvironment
from .evaluator import (
    EpisodeLLMEvaluator,
    EvaluationDimension,
    EvaluationForMultipleAgents,
    EvaluationForTwoAgents,
    Evaluator,
    RuleBasedTerminatedEvaluator,
    TinyChatDimensions,
    unweighted_aggregate_evaluate,
)
from .messages import AgentAction, ChatBackground, Message, Observation, SimpleMessage
from .profile import BaseAgentProfile, BaseEnvironmentProfile, BaseRelationshipProfile
from .server import ChatServer

__all__ = [
    'LLMAgent',
    'Message',
    'SimpleMessage',
    'Observation',
    'AgentAction',
    'ChatBackground',
    'TinyChatEnvironment',
    'BaseAgentProfile',
    'BaseEnvironmentProfile',
    'BaseRelationshipProfile',
    'Evaluator',
    'RuleBasedTerminatedEvaluator',
    'EpisodeLLMEvaluator',
    'TinyChatDimensions',
    'EvaluationDimension',
    'EvaluationForTwoAgents',
    'EvaluationForMultipleAgents',
    'unweighted_aggregate_evaluate',
    'ChatServer',
]
