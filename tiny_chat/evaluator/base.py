import abc
from typing import Any

from tiny_chat.messages import Message


class BaseEvaluator(abc.ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @property
    @abc.abstractmethod
    def evaluator_type(self) -> str:
        pass

    @abc.abstractmethod
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError
