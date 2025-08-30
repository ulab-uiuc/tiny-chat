import logging
from collections.abc import Callable
from typing import Any

from pydantic import validate_call

from tiny_chat.messages import AgentAction, Message

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class RuleBasedEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: dict[str, Any] = None,
        max_turn_number: int = 20,
        max_stale_turn: int = 2,
        leave_detector: Callable[[list[tuple[str, Message]]], bool] | None = None,
    ) -> None:
        if config is not None:
            super().__init__(config)
            self.max_turn_number = config.get('max_turn_number', max_turn_number)
            self.max_stale_turn = config.get('max_stale_turn', max_stale_turn)
            self.leave_detector = config.get('leave_detector', leave_detector)
        else:
            super().__init__({})
            self.max_turn_number = max_turn_number
            self.max_stale_turn = max_stale_turn
            self.leave_detector = leave_detector

    @property
    def evaluator_type(self) -> str:
        return 'rule_based'

    @validate_call
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number
        # Rule 2: If one of the players leaves, terminate the conversation (default)
        if self.leave_detector is not None:
            someone_leaving = bool(self.leave_detector(messages))
        else:
            someone_leaving = False
            for source, msg in messages[::-1]:
                if source == 'Environment':
                    continue
                if isinstance(msg, AgentAction) and msg.action_type == 'leave':
                    someone_leaving = True
                    break
        # Rule 3: If the conversation is stale for too long, terminate the conversation
        stale_count = 0
        for message in messages[::-1]:
            if message[0] == 'Environment':
                continue
            assert isinstance(message[1], AgentAction)
            if message[1].action_type == 'none':
                stale_count += 1
            else:
                break
            if stale_count > self.max_stale_turn:
                break
        stale_too_long = stale_count > self.max_stale_turn
        terminated = conversation_too_long or someone_leaving or stale_too_long
        reasons_for_termination = (
            f'{"The conversation is too long; " if conversation_too_long else ""}'
            f'{"Someone is leaving; " if someone_leaving else ""}'
            f'{"The conversation stales for too long; " if stale_too_long else ""}'
        )
        return [
            (
                'environment',
                (('terminated', terminated), reasons_for_termination),
            )
        ]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self(turn_number, messages)
