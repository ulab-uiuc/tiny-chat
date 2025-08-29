import random
from typing import Any, Generator, Type, TypeVar
from ..data_loader import DataLoader
from tiny_chat.agents import BaseAgent
from tiny_chat.envs import TinyChatEnvironment
from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class UniformSampler(BaseSampler[ObsType, ActType]):
    def sample(self,
               agent_classes: Type[BaseAgent[ObsType, ActType]] | list[Type[BaseAgent[ObsType, ActType]]],
               agent_num: int = 2,
               replacement: bool = True,
               size: int = 1,
               env_params: dict[str, Any] = {},
               agents_params: list[dict[str, Any]] = [{}, {}]) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        # check agent_classes
        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * agent_num
        elif len(agent_classes) != agent_num:
            raise ValueError("Length of agent_classes must match agent_num")
        
        if len(agents_params) != agent_num:
            raise ValueError("Length of agents_params must match agent_num")

        # Load profiles if not provided, use official dataset
        data_loader = DataLoader()
        if self.agent_list is None:
            self.agent_list = data_loader.get_all_agent_profiles()
        if self.env_list is None:
            self.env_list = data_loader.get_all_env_profiles()

        # set up environment
        for _ in range(size):
            env_profile = random.choice(self.env_list)
            env = TinyChatEnvironment(**env_params)

            # Sample agent profiles
            if len(self.agent_list) < agent_num:
                raise ValueError("Not enough agent profiles")
            sampled_agent_profiles = random.sample(self.agent_list, agent_num)
            
            agents = []
            for agent_class, agent_profile, agent_params in zip(agent_classes, sampled_agent_profiles, agents_params):
                agent = agent_class(agent_profile=agent_profile, **agent_params)
                agents.append(agent)
                
            # set goal for each agent
            for agent, goal in zip(agents, env_profile.agent_goals):
                agent.goal = goal

            yield (env, agents)