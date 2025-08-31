import ast
import random
from typing import Any, Generator, Type, TypeVar
from ..data_loader import DataLoader
from tiny_chat.agents import BaseAgent
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.profiles import BaseAgentProfile, BaseEnvironmentProfile, BaseRelationshipProfile
from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ConstraintSampler(BaseSampler[ObsType, ActType]):
    """A sampler that use the constraints in environment_profile"""
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
        
        # only support 2 agents for now
        if agent_num != 2:
            raise NotImplementedError("Only support 2 agents for now")
        
        # Load profiles if not provided, use official dataset
        data_loader = DataLoader()
        if self.agent_list is None:
            self.agent_list = data_loader.get_all_agent_profiles()
        if self.env_list is None:
            self.env_list = data_loader.get_all_env_profiles()
        relationships = data_loader.get_all_relationship_profiles()

        if not replacement:
            # pick one environment profile
            env_profile = random.choice(self.env_list)
            env = TinyChatEnvironment(**env_params)

            # find agents that fullfill the constraints in env_profile
            sampled_agent_pairs = self.find_agent_pairs(env_profile, relationships)
            random.shuffle(sampled_agent_pairs)

            for i in range(size):
                if len(sampled_agent_pairs) < size:
                    raise ValueError("No agent pairs found that satisfy the constraints")
                sampled_agent_profiles = sampled_agent_pairs[i]

                agents = []
                for agent_class, agent_profile, agent_params in zip(agent_classes, sampled_agent_profiles, agents_params):
                    agent = agent_class(agent_profile=agent_profile, **agent_params)
                    agents.append(agent)
                    
                # set goal for each agent
                for agent, goal in zip(agents, env_profile.agent_goals):
                    agent.goal = goal
            
                yield (env, agents)
        
        else:
            for _ in range(size):
                # pick one environment profile
                env_profile = random.choice(self.env_list)
                env = TinyChatEnvironment(**env_params)

                # find agents that fullfill the constraints in env_profile
                sampled_agent_pairs = self.find_agent_pairs(env_profile, relationships)
                if len(sampled_agent_pairs) == 0:
                    raise ValueError("No agent pairs found that satisfy the constraints")
                sampled_agent_profiles = random.choice(sampled_agent_pairs)
                
                agents = []
                for agent_class, agent_profile, agent_params in zip(agent_classes, sampled_agent_profiles, agents_params):
                    agent = agent_class(agent_profile=agent_profile, **agent_params)
                    agents.append(agent)
                    
                # set goal for each agent
                for agent, goal in zip(agents, env_profile.agent_goals):
                    agent.goal = goal

                yield (env, agents)


    def find_agent_pairs(self, 
                    env: BaseEnvironmentProfile, 
                    relationships: BaseRelationshipProfile) -> list[list[BaseAgentProfile]]:
        sampled_agent_pairs = []
        for rel in relationships:
            agent1 = self.get_agent(rel.agent_ids[0])
            agent2 = self.get_agent(rel.agent_ids[1])
            if agent1 is None or agent2 is None:
                continue
            # check the age constraint
            if env.age_constraint and env.age_constraint != "[(18, 70), (18, 70)]":
                age_contraint = ast.literal_eval(env.age_contraint)
                if not (age_contraint[0][0] <= agent1.age <= age_contraint[0][1] and 
                        age_contraint[1][0] <= agent2.age <= age_contraint[1][1]):
                    continue
            # check the occupation constraint
            if env.occupation_constraint and env.occupation_constraint != "nan" and env.occupation_constraint != "[[], []]":
                occupation_constraint = ast.literal_eval(env.occupation_constraint)
                if not (agent1.occupation.lower() in occupation_constraint[0] and
                        agent2.occupation.lower() in occupation_constraint[1]):
                    continue
            # check agent constraint: not supported yet            

            # add the pair
            sampled_agent_pairs.append([agent1, agent2])

        return sampled_agent_pairs

    
    def get_agent(self, agent_id: str) -> BaseAgentProfile | None:
        for agent in self.agent_list:
            if agent.pk == agent_id:
                return agent
        return None