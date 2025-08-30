from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import Any, Generic, TypeVar

from tiny_chat.agents import BaseAgent
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.profiles import BaseAgentProfile, BaseEnvironmentProfile

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

EnvAgentCombo = tuple[TinyChatEnvironment, Sequence[BaseAgent[ObsType, ActType]]]


class BaseSampler(Generic[ObsType, ActType], ABC):
    """Base class for sampling environments and agents."""

    def __init__(
        self,
        env_candidates: Sequence[BaseEnvironmentProfile | str] | None = None,
        agent_candidates: Sequence[BaseAgentProfile | str] | None = None,
    ) -> None:
        """Initialize the sampler with candidate pools.

        Args:
            env_candidates: Pool of environment profiles to sample from
            agent_candidates: Pool of agent profiles to sample from
        """
        self.env_candidates = env_candidates
        self.agent_candidates = agent_candidates

    @abstractmethod
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
        """Sample an environment and a list of agents.

        Args:
            agent_classes: A single agent class for all sampled agents or a list of agent classes
            n_agent: Number of agents. Defaults to 2
            replacement: Whether to sample with replacement. Defaults to True
            size: Number of samples. Defaults to 1
            env_params: Parameters for the environment. Defaults to {}
            agents_params: Parameters for the agents. Defaults to [{}, {}]

        Returns:
            Generator yielding tuples of (TinyChatEnvironment, list[BaseAgent])
        """
        raise NotImplementedError
