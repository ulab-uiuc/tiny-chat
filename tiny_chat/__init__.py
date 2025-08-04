"""
Tiny Chat - A simplified multi-agent chat system inspired by Sotopia
"""

__version__ = "0.0.1"

from .agents import LLMAgent
from .messages import Message, SimpleMessage, Observation, AgentAction, ChatBackground
from .environment import ChatEnvironment
from .generator import MessageGenerator
from .profile import AgentProfile, RelationshipProfile, RelationshipType
from .evaluators import (
    Evaluator,
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    SotopiaDimensions,
    EvaluationDimension,
    EvaluationForTwoAgents,
    EvaluationForMultipleAgents,
    unweighted_aggregate_evaluate,
)
from .server import ChatServer

__all__ = [
    "LLMAgent",
    "Message",
    "SimpleMessage",
    "Observation",
    "AgentAction",
    "ChatBackground",
    "ChatEnvironment",
    "MessageGenerator",
    "AgentProfile",
    "RelationshipProfile",
    "RelationshipType",
    "Evaluator",
    "RuleBasedTerminatedEvaluator",
    "EpisodeLLMEvaluator",
    "SotopiaDimensions",
    "EvaluationDimension",
    "EvaluationForTwoAgents",
    "EvaluationForMultipleAgents",
    "unweighted_aggregate_evaluate",
    "ChatServer",
]
