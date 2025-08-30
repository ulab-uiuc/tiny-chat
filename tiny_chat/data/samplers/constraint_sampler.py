import ast
import random
from collections.abc import Generator
from typing import Any, TypeVar

from tiny_chat.agents import BaseAgent
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.profiles import (
    BaseAgentProfile,
    BaseEnvironmentProfile,
    BaseRelationshipProfile,
)

from ..loader import DataLoader
from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


def _get_fit_agents_for_one_env(
    env_profile: BaseEnvironmentProfile,
    agent_candidates: list[BaseAgentProfile],
    relationship_candidates: list[BaseRelationshipProfile],
    size: int,
) -> list[list[BaseAgentProfile]]:
    """Get agents that fit the constraints of one environment."""

    relationship_constraint = env_profile.relationship

    available_relationships = [
        rel
        for rel in relationship_candidates
        if rel.default_relationship == relationship_constraint
    ]

    age_constraint = env_profile.age_constraint
    if age_constraint and age_constraint != '[(18, 70), (18, 70)]':
        try:
            age_constraint_list = ast.literal_eval(age_constraint)
            filtered_relationships = []

            for relationship in available_relationships:
                if len(relationship.agent_ids) >= 2:
                    agent1_id, agent2_id = relationship.agent_ids[:2]

                    agent1 = next(
                        (a for a in agent_candidates if a.pk == agent1_id), None
                    )
                    agent2 = next(
                        (a for a in agent_candidates if a.pk == agent2_id), None
                    )

                    if agent1 and agent2:
                        if (
                            age_constraint_list[0][0]
                            <= agent1.age
                            <= age_constraint_list[0][1]
                            and age_constraint_list[1][0]
                            <= agent2.age
                            <= age_constraint_list[1][1]
                        ):
                            filtered_relationships.append(relationship)

            available_relationships = filtered_relationships
        except (ValueError, SyntaxError):
            pass

    if len(available_relationships) < size:
        raise ValueError(
            f'Number of available relationships ({len(available_relationships)}) '
            f'is smaller than the required size ({size})'
        )

    random.shuffle(available_relationships)
    selected_relationships = available_relationships[:size]

    fit_agents = []
    for relationship in selected_relationships:
        if len(relationship.agent_ids) >= 2:
            agent1_id, agent2_id = relationship.agent_ids[:2]

            agent1 = next((a for a in agent_candidates if a.pk == agent1_id), None)
            agent2 = next((a for a in agent_candidates if a.pk == agent2_id), None)

            if agent1 and agent2:
                fit_agents.append([agent1, agent2])

    return fit_agents


class ConstraintBasedSampler(BaseSampler[ObsType, ActType]):
    """Constraint-based sampler that respects environment constraints."""

    def __init__(
        self,
        env_candidates: list[BaseEnvironmentProfile | str] | None = None,
        agent_candidates: list[BaseAgentProfile | str] | None = None,
        relationship_candidates: list[BaseRelationshipProfile | str] | None = None,
        data_loader: DataLoader | None = None,
    ) -> None:
        """Initialize constraint-based sampler.

        Args:
            env_candidates: Pool of environment profiles to sample from
            agent_candidates: Pool of agent profiles to sample from
            relationship_candidates: Pool of relationship profiles to sample from
            data_loader: DataLoader instance for loading profiles from datasets
        """
        super().__init__(env_candidates, agent_candidates)
        self.relationship_candidates = relationship_candidates
        self.data_loader = data_loader or DataLoader()

    def sample(
        self,
        agent_classes: (
            type[BaseAgent[ObsType, ActType]] | list[type[BaseAgent[ObsType, ActType]]]
        ),
        n_agent: int = 2,
        replacement: bool = True,
        size: int = 5,
        env_params: dict[str, Any] = {},
        agents_params: list[dict[str, Any]] = [{}, {}],
    ) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        """
        Sample an environment and a list of agents based on the constraints of the environment.

        Note: Sampling without replacement is only restricted to single env candidate.
        This is due to the fact that the number of possible combinations of env and agents is huge.
        Please sample for each env separately if you want to sample without replacement.
        """
        assert (
            not isinstance(agent_classes, list) or len(agent_classes) == n_agent
        ), f'agent_classes should be a list of length {n_agent} or a single agent class'

        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * n_agent

        assert (
            len(agents_params) == n_agent
        ), f'agents_params should be a list of length {n_agent}'

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

        if self.relationship_candidates is None:
            relationship_candidates = self.data_loader.get_all_relationship_profiles()
            if not relationship_candidates:
                raise ValueError('No relationship candidates available for sampling.')
            self.relationship_candidates = relationship_candidates

        env_profiles: list[BaseEnvironmentProfile] = []
        agent_profiles_list: list[list[BaseAgentProfile]] = []

        if not replacement:
            assert len(self.env_candidates) == 1, (
                'Sampling without replacement is only restricted to single env candidate. '
                'This is due to the fact that the number of possible combinations of env and agents is huge. '
                'Please sample for each env separately if you want to sample without replacement.'
            )

            env_profile = self.env_candidates[0]
            if isinstance(env_profile, str):
                raise NotImplementedError('String environment IDs not yet supported')

            agents_which_fit_scenario = _get_fit_agents_for_one_env(
                env_profile, self.agent_candidates, self.relationship_candidates, size
            )
            env_profiles = [env_profile] * size
            agent_profiles_list = agents_which_fit_scenario
        else:
            for _ in range(size):
                env_profile = random.choice(self.env_candidates)
                if isinstance(env_profile, str):
                    raise NotImplementedError(
                        'String environment IDs not yet supported'
                    )

                env_profiles.append(env_profile)
                agents_which_fit_scenario = _get_fit_agents_for_one_env(
                    env_profile, self.agent_candidates, self.relationship_candidates, 1
                )
                agent_profiles_list.append(agents_which_fit_scenario[0])

        assert len(env_profiles) == size, 'Number of env_profiles is not equal to size'
        assert (
            len(agent_profiles_list) == size
        ), 'Number of agent_profiles_list is not equal to size'

        for env_profile, agent_profile_list in zip(
            env_profiles, agent_profiles_list, strict=False
        ):
            env = TinyChatEnvironment(**env_params)

            agents = [
                agent_class(agent_profile=agent_profile, **agent_params)
                for agent_class, agent_profile, agent_params in zip(
                    agent_classes, agent_profile_list, agents_params, strict=False
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
