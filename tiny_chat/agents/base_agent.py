import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from tiny_chat.messages import MessengerMixin
from tiny_chat.profile import BaseAgentProfile

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class BaseAgent(Generic[ObsType, ActType], MessengerMixin, ABC):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: BaseAgentProfile | None = None,
        profile_jsonl_path: str | None = None,
    ) -> None:
        MessengerMixin.__init__(self)

        if agent_profile is not None:
            self.profile = agent_profile
            self.agent_name = self.profile.first_name + ' ' + self.profile.last_name

        elif uuid_str is not None:
            if not profile_jsonl_path:
                raise ValueError('uuid_str provided but profile_jsonl_path is missing.')
            profile = self._load_profile_from_jsonl(profile_jsonl_path, uuid_str)
            if profile is None:
                raise ValueError(
                    f"Agent with uuid '{uuid_str}' not found in {profile_jsonl_path}"
                )
            if isinstance(profile, dict):
                self.profile = BaseAgentProfile(**profile)
            else:
                self.profile = profile
            self.agent_name = (
                f'{self.profile.first_name} {self.profile.last_name}'.strip()
            )

        elif agent_name is not None:
            self.profile = None  # type: ignore
            self.agent_name = agent_name

        else:
            raise ValueError(
                'Either agent_profile, uuid_str, or agent_name must be provided.'
            )

        self._goal: str | None = None
        self.model_name: str = ''

    @property
    def goal(self) -> str:
        assert self._goal is not None, 'attribute goal has to be set before use'
        return self._goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    @abstractmethod
    async def act(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    def reset(self) -> None:
        self.reset_inbox()
        self._goal = None

    @staticmethod
    def _load_profile_from_jsonl(path: str, uuid_str: str) -> dict[str, Any] | None:
        with open(path, encoding='utf-8') as f:
            for line in f:
                data: dict[str, Any] = json.loads(line)
                if data.get('uuid') == uuid_str:
                    return data
        return None
