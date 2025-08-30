from abc import ABC, abstractmethod
from typing import Any

from tiny_chat.messages import AgentAction, Observation


class BaseChatEnivronment(ABC):
    @abstractmethod
    def reset(self, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def step(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> Any:
        pass

    @abstractmethod
    def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> Any:
        pass

    @abstractmethod
    def get_observation(self, agent_name: str) -> Observation:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @abstractmethod
    def get_turn_number(self) -> int:
        pass
