import random
from typing import Any, Generator, Generic, Sequence, Type, TypeVar
from ..data_loader import DataLoader
from tiny_chat.agents import BaseAgent
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.profiles import BaseAgentProfile, BaseEnvironmentProfile, BaseRelationshipProfile

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
EnvAgentCombo = tuple[TinyChatEnvironment, Sequence[BaseAgent[ObsType, ActType]]]


class BaseSampler:
    def __init__(self, 
                 agent_list: Sequence[BaseAgentProfile | str] | None = None,
                 env_list: Sequence[BaseEnvironmentProfile | str] | None = None,):
        self.agent_list = agent_list
        self.env_list = env_list

    
    def sample(self, 
               agent_classes: Type[BaseAgent[ObsType, ActType]] | list[Type[BaseAgent[ObsType, ActType]]],
               agent_num: int = 2,
               replacement: bool = True,
               size: int = 1,
               env_params: dict[str, Any] = {},
               agents_params: list[dict[str, Any]] = [{}, {}]) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        raise NotImplementedError