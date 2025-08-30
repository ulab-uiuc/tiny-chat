import random
from collections.abc import Generator
from typing import Any, TypeVar

from tiny_chat.agents import BaseAgent
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.profiles import BaseAgentProfile, BaseEnvironmentProfile

from ..loader import DataLoader
from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class UniformSampler(BaseSampler[ObsType, ActType]):
    """Uniform random sampler for environments and agents."""

    def __init__(
        self,
        env_candidates: list[BaseEnvironmentProfile | str] | None = None,
        agent_candidates: list[BaseAgentProfile | str] | None = None,
        data_loader: DataLoader | None = None,
    ) -> None:
        """Initialize uniform sampler.

        Args:
            env_candidates: Pool of environment profiles to sample from
            agent_candidates: Pool of agent profiles to sample from
            data_loader: DataLoader instance for loading profiles from datasets
        """
        super().__init__(env_candidates, agent_candidates)
        self.data_loader = data_loader or DataLoader()

    def sample(
        self,
        agent_classes: (
            type[BaseAgent[ObsType, ActType]] | list[type[BaseAgent[ObsType, ActType]]]
        ),
        n_agent: int = 2,
        replacement: bool = True,
        size: int = 1,
        env_params: dict[str, Any] = {},
        agents_params: list[dict[str, Any]] = [{}, {}],
    ) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        """
        Sample an environment and `n_agent` agents uniformly at random.

        Runtime checks:
        1. If `agent_classes` is a list, it should have length `n_agent`.
        2. `agents_params` should also be a list of length `n_agent`.

        Note: Currently, uniform sampling without replacement is not supported.
        This is due to the difficulty of sequentially sampling environment and agents.
        In theory, we can reject samples that have been sampled before, but this is not efficient.
        Please open an issue if you need this feature.
        """
        assert (
            not isinstance(agent_classes, list) or len(agent_classes) == n_agent
        ), f'agent_classes should be a list of length {n_agent} or a single agent class'

        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * n_agent

        assert (
            len(agents_params) == n_agent
        ), f'agents_params should be a list of length {n_agent}'

        assert replacement, 'Uniform sampling without replacement is not supported yet'

        if self.env_candidates is None:
            env_candidates = self.data_loader.get_all_env_profiles()
            if not env_candidates:
                raise ValueError('No environment candidates available for sampling.')
            self.env_candidates = env_candidates

        if self.agent_candidates is None:
            agent_candidates = self.data_loader.get_all_agent_profiles()
            if not agent_candidates:
                raise ValueError('No agent candidates available for sampling.')
            self.agent_candidates = agent_candidates

        for _ in range(size):
            env_profile = random.choice(self.env_candidates)
            if isinstance(env_profile, str):
                raise NotImplementedError('String environment IDs not yet supported')

            env = TinyChatEnvironment(**env_params)

            agent_profile_candidates = self.agent_candidates
            if len(agent_profile_candidates) == n_agent:
                agent_profiles_maybe_id = agent_profile_candidates
            else:
                agent_profiles_maybe_id = random.sample(
                    agent_profile_candidates, n_agent
                )

            agent_profiles = []
            for i in agent_profiles_maybe_id:
                if isinstance(i, BaseAgentProfile):
                    agent_profiles.append(i)
                else:
                    raise NotImplementedError('String agent IDs not yet supported')

            agents = [
                agent_class(agent_profile=agent_profile, **agent_params)
                for agent_class, agent_profile, agent_params in zip(
                    agent_classes, agent_profiles, agents_params, strict=False
                )
            ]

            if hasattr(env_profile, 'agent_goals') and env_profile.agent_goals:
                for agent, goal in zip(agents, env_profile.agent_goals, strict=False):
                    agent.goal = goal
            else:
                for i, agent in enumerate(agents):
                    agent.goal = (
                        f'Participate effectively in this conversation as agent {i+1}'
                    )

            yield env, agents
