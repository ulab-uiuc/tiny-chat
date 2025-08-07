import asyncio
import copy
import itertools
import random
from typing import Any, Literal, TypeVar

from pydantic import validate_call

from tiny_chat.evaluator import Evaluator, unweighted_aggregate_evaluate
from tiny_chat.messages import (
    ActionType,
    AgentAction,
    Message,
    MultiAgentChatBackground,
    Observation,
    ScriptBackground,
    SimpleMessage,
)

TBackground = TypeVar('TBackground', bound=ScriptBackground)

# Default action types available to agents
DEFAULT_ACTION_TYPES: set[ActionType] = {
    'none',
    'speak',
    'non-verbal communication',
    'action',
    'leave',
}


def _actions_to_natural_language(actions: dict[str, AgentAction]) -> str:
    action_str = ''
    for agent, action in actions.items():
        # Only record actions that did something
        if action.action_type != 'none':
            if action_str != '':
                action_str += ';'  # separate actions with semicolon
            action_str += f'{agent} {action.to_natural_language()}'
    return action_str


class BaseChatEnivronment:
    def reset(self, **kwargs):
        pass

    def step(self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]):
        pass

    def get_observation(self, agent_name: str) -> Observation:
        pass

    def is_terminated(self) -> bool:
        pass

    def get_turn_number(self) -> int:
        pass


class TwoAgentTinyChatEnvironment(BaseChatEnivronment):
    def __init__(
        self,
        available_action_types: set[ActionType] = DEFAULT_ACTION_TYPES,
        action_order: Literal['simultaneous', 'round-robin', 'random'] = 'simultaneous',
        evaluators: list[Evaluator] | None = None,
        model_name: str = 'gpt-4o-mini',
        terminal_evaluators: list[Evaluator] | None = None,
        background_class: type[TBackground] | None = None,
    ) -> None:
        """A chat environment for parallel agents.

        Args:
            available_action_types (set[ActionType], optional): The action types that are available to the agents. Defaults to set(["none", "speak", "non-verbal communication", "action"]).
            action_order (Literal["simultaneous", "round-robin", "random"], optional): The order in which the agents take actions. Defaults to "simultaneous".
            model_name (str, optional): The name of the language model to use. Defaults to "gpt-4o-mini".
        """
        super().__init__()
        if background_class is None:
            self.background_class = ScriptBackground
        else:
            self.background_class = background_class
        self.background = self.background_class(
            scenario='',
            p1_background='',
            p2_background='',
            p1_goal='',
            p2_goal='',
            p1_name='',
            p2_name='',
        )

        self.agents = []
        self.action_spaces = {}
        self.available_action_types = list(available_action_types)
        self.action_order = action_order
        self.action_mask: list[bool] = []
        self.evaluators = evaluators or []
        self.terminal_evaluators = terminal_evaluators or []
        self.model_name = model_name
        self.turn_number = 0
        self.inbox: list[tuple[str, Message]] = []

    def reset_inbox(self) -> None:
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: dict[str, Any] | None = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> dict[str, Observation]:
        """Starting a new episode. Must be called before step().

        Args:
            seed (int, optional): Seed for the environment. Defaults to None. Not used right now.
            options (dict, optional): Options for the environment. Defaults to None.
            agents (dict, optional): Dictionary of agents with their profiles. Defaults to None.
            omniscient (bool, optional): Whether the agents know the other agent's goal. Defaults to False.
            lite (bool, optional): Whether to use lite mode (no backgrounds). Defaults to False.
        """
        self.reset_inbox()

        if agents is not None:
            assert agents, 'agents must be provided'
            # assert len(agents) == 2, 'Only supporting two agents right now'
            agent_names = list(agents.keys())

            # Extract goals from agents if available
            agent_goals = []
            for agent_name in agent_names:
                agent = agents[agent_name]
                if hasattr(agent, 'goal'):
                    agent_goals.append(agent.goal)
                else:
                    agent_goals.append('Achieve your objectives in this conversation')

            raw_background = self.background_class(
                scenario=(
                    options.get('scenario', 'A conversation between two agents')
                    if options
                    else 'A conversation between two agents'
                ),
                p1_background=(
                    self._get_agent_background(agents[agent_names[0]], 0)
                    if not lite
                    else ''
                ),
                p2_background=(
                    self._get_agent_background(agents[agent_names[1]], 1)
                    if not lite
                    else ''
                ),
                p1_goal=f"<root viewer='agent_0'>{agent_goals[0]}</root>",
                p2_goal=f"<root viewer='agent_1'>{agent_goals[1]}</root>",
                p1_name=agent_names[0],
                p2_name=agent_names[1],
            )

            self.background = self.background_class(
                scenario=raw_background.scenario,
                p1_background=raw_background.p1_background,
                p2_background=raw_background.p2_background,
                p1_goal=raw_background.p1_goal,
                p2_goal=raw_background.p2_goal,
                p1_name=raw_background.p1_name,
                p2_name=raw_background.p2_name,
            )
        else:
            raise ValueError('agents must be provided')

        self.agents = [self.background.p1_name, self.background.p2_name]
        agent_backgrounds = []
        if omniscient:
            for _ in range(self.num_agents):
                agent_backgrounds.append(copy.deepcopy(self.background))
        else:
            for _ in range(self.num_agents):
                agent_backgrounds.append(
                    self.background_class(
                        scenario=raw_background.scenario,
                        p1_background=raw_background.p1_background,
                        p2_background=raw_background.p2_background,
                        p1_goal=raw_background.p1_goal,
                        p2_goal=raw_background.p2_goal,
                        p1_name=raw_background.p1_name,
                        p2_name=raw_background.p2_name,
                    )
                )
        background_for_a = agent_backgrounds[0]
        background_for_b = agent_backgrounds[1]

        if not omniscient:
            background_for_a.p2_goal = 'Unknown'
            background_for_b.p1_goal = 'Unknown'

        self.action_spaces = {
            agent: {
                'action_type': list(range(len(self.available_action_types))),
                'argument': str,
            }
            for agent in self.agents
        }
        self.turn_number = 0
        self.action_mask = [False for _ in self.agents]
        if self.action_order == 'round-robin':
            self.action_mask[0] = True
        elif self.action_order == 'random':
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]

        self.recv_message('Environment', self.background)

        return {
            self.background.p1_name: Observation(
                last_turn=background_for_a.to_natural_language(),
                turn_number=0,
                available_actions=(
                    list(self.available_action_types)
                    if self.action_mask[0]
                    else ['none']
                ),
            ),
            self.background.p2_name: Observation(
                last_turn=background_for_b.to_natural_language(),
                turn_number=0,
                available_actions=(
                    list(self.available_action_types)
                    if self.action_mask[1]
                    else ['none']
                ),
            ),
        }

    def get_turn_number(self) -> int:
        return self.turn_number

    def is_terminated(self) -> bool:
        return self.turn_number >= self.max_turns

    def get_observation(self, agent_name: str) -> Observation:
        # get last turn
        last_turn = ''
        if self.turn_number > 0 and self.inbox:
            last_actions = {}
            for source, message in self.inbox:
                if isinstance(message, AgentAction) and source != 'Environment':
                    last_actions[source] = message
            if last_actions:
                last_turn = _actions_to_natural_language(last_actions)
            else:
                last_turn = self.background.to_natural_language()
        else:
            last_turn = self.background.to_natural_language()

        agent_index = 0
        if hasattr(self, 'agents') and self.agents:
            try:
                agent_index = self.agents.index(agent_name)
            except ValueError:
                agent_index = 0

        available_actions = ['none']
        if hasattr(self, 'action_mask') and len(self.action_mask) > agent_index:
            if self.action_mask[agent_index]:
                available_actions = list(self.available_action_types)

        obs = Observation(
            last_turn=last_turn,
            turn_number=self.turn_number,
            available_actions=available_actions,
        )
        return obs

    def _get_agent_background(self, agent: Any, agent_id: int) -> str:
        """Extract background information from agent"""
        if hasattr(agent, 'profile'):
            profile = agent.profile
            if hasattr(profile, 'to_background_string'):
                return profile.to_background_string(agent_id)
            elif hasattr(profile, 'description'):
                return f"<root><p viewer='agent_{agent_id}'>{profile.description}</p></root>"
        elif hasattr(agent, 'description'):
            return f"<root><p viewer='agent_{agent_id}'>{agent.description}</p></root>"
        return f"<root><p viewer='agent_{agent_id}'>Agent {agent_id}</p></root>"

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @validate_call
    def step(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        # Time step ++
        self.turn_number += 1

        # For action sampled from action space, it needs to be converted into AgentAction
        complied_actions: dict[str, AgentAction] = {}
        for key in actions.keys():
            action = actions[key]
            if isinstance(action, AgentAction):
                complied_actions[key] = action
            else:
                action['action_type'] = self.available_action_types[
                    int(action['action_type'])
                ]
                complied_actions[key] = AgentAction.parse_obj(action)

        # Masking actions from agent that are in turn
        for idx, agent in enumerate(self.agents):
            if not self.action_mask[idx]:
                complied_actions[agent] = AgentAction(action_type='none', argument='')

        self.recv_message(
            'Environment', SimpleMessage(message=f'Turn #{self.turn_number}')
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

        response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *(
                        evaluator(turn_number=self.turn_number, messages=self.inbox)
                        for evaluator in self.evaluators
                    )
                )
            )
        )

        self.action_mask = [False for _ in self.agents]
        if self.action_order == 'round-robin':
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == 'random':
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]
        obs = _actions_to_natural_language(complied_actions)
        return (
            {
                self.background.p1_name: Observation(
                    last_turn=obs,
                    turn_number=self.turn_number,
                    available_actions=(
                        list(self.available_action_types)
                        if self.action_mask[0]
                        else ['none']
                    ),
                ),
                self.background.p2_name: Observation(
                    last_turn=obs,
                    turn_number=self.turn_number,
                    available_actions=(
                        list(self.available_action_types)
                        if self.action_mask[1]
                        else ['none']
                    ),
                ),
            },
            {
                self.background.p1_name: (
                    (
                        response.p1_rate
                        if isinstance(response.p1_rate, float)
                        else response.p1_rate[0]
                    )
                    if response.p1_rate
                    else 0
                ),
                self.background.p2_name: (
                    (
                        response.p2_rate
                        if isinstance(response.p2_rate, float)
                        else response.p2_rate[0]
                    )
                    if response.p2_rate
                    else 0
                ),
            },
            {
                self.background.p1_name: response.terminated,
                self.background.p2_name: response.terminated,
            },
            {
                self.background.p1_name: False,
                self.background.p2_name: False,
            },
            {
                self.background.p1_name: {
                    'comments': response.comments or '',
                    'complete_rating': response.p1_rate or 0,
                },
                self.background.p2_name: {
                    'comments': response.comments or '',
                    'complete_rating': response.p2_rate or 0,
                },
            },
        )

    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        self.turn_number += 1

        complied_actions = self._parse_actions(actions)
        self._mask_actions(complied_actions)
        self._broadcast_actions(complied_actions)

        response = await self._evaluate_turn()

        if response.terminated:
            response = await self._handle_termination(response)

        self._update_action_mask()

        obs = _actions_to_natural_language(complied_actions)
        info = self._build_info(response)

        return (
            self._build_observations(obs),
            self._build_rewards(response),
            self._build_done_flags(response),
            self._build_truncation_flags(),
            info,
        )

    def _parse_actions(self, actions: dict[str, Any]) -> dict[str, AgentAction]:
        result = {}
        for key, action in actions.items():
            if isinstance(action, AgentAction):
                result[key] = action
            else:
                action['action_type'] = self.available_action_types[
                    int(action['action_type'])
                ]
                result[key] = AgentAction.parse_obj(action)
        return result

    def _mask_actions(self, actions: dict[str, AgentAction]) -> None:
        for idx, agent in enumerate(self.agents):
            if not self.action_mask[idx]:
                actions[agent] = AgentAction(action_type='none', argument='')

    def _broadcast_actions(self, actions: dict[str, AgentAction]) -> None:
        self.recv_message(
            'Environment', SimpleMessage(message=f'Turn #{self.turn_number}')
        )
        for agent, action in actions.items():
            self.recv_message(agent, action)

    async def _evaluate_turn(self) -> Any:
        return unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *await asyncio.gather(
                        *[
                            evaluator.__acall__(
                                turn_number=self.turn_number, messages=self.inbox
                            )
                            for evaluator in self.evaluators
                        ]
                    )
                )
            )
        )

    async def _handle_termination(self, response):
        terminal_response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *await asyncio.gather(
                        *[
                            evaluator.__acall__(
                                turn_number=self.turn_number, messages=self.inbox
                            )
                            for evaluator in self.terminal_evaluators
                        ]
                    )
                )
            )
        )
        response.p1_rate = response.p1_rate or terminal_response.p1_rate
        response.p2_rate = response.p2_rate or terminal_response.p2_rate
        if response.comments and terminal_response.comments:
            response.comments += terminal_response.comments
        elif terminal_response.comments:
            response.comments = terminal_response.comments
        return response

    def _update_action_mask(self):
        self.action_mask = [False for _ in self.agents]
        if self.action_order == 'round-robin':
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == 'random':
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]

    def _build_info(self, response) -> dict[str, Any]:
        info = {
            self.background.p1_name: {
                'comments': response.comments or '',
                'complete_rating': response.p1_rate or 0,
            },
            self.background.p2_name: {
                'comments': response.comments or '',
                'complete_rating': response.p2_rate or 0,
            },
        }
        if response.terminated and self.terminal_evaluators:
            info['rewards_prompt'] = {
                'overall_prompt': self.terminal_evaluators[0].prompt  # type: ignore
            }
        return info

    def _build_observations(self, obs: str) -> dict[str, Observation]:
        return {
            self.background.p1_name: Observation(
                last_turn=obs,
                turn_number=self.turn_number,
                available_actions=(
                    list(self.available_action_types)
                    if self.action_mask[0]
                    else ['none']
                ),
            ),
            self.background.p2_name: Observation(
                last_turn=obs,
                turn_number=self.turn_number,
                available_actions=(
                    list(self.available_action_types)
                    if self.action_mask[1]
                    else ['none']
                ),
            ),
        }

    def _build_rewards(self, response) -> dict[str, float]:
        return {
            self.background.p1_name: (
                response.p1_rate[0]
                if isinstance(response.p1_rate, list)
                else response.p1_rate
            )
            or 0,
            self.background.p2_name: (
                response.p2_rate[0]
                if isinstance(response.p2_rate, list)
                else response.p2_rate
            )
            or 0,
        }

    def _build_done_flags(self, response) -> dict[str, bool]:
        return {
            self.background.p1_name: response.terminated,
            self.background.p2_name: response.terminated,
        }

    def _build_truncation_flags(self) -> dict[str, bool]:
        return {
            self.background.p1_name: False,
            self.background.p2_name: False,
        }

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        pass


class MultiAgentTinyChatEnvironment(BaseChatEnivronment):
    def __init__(
        self,
        available_action_types: set[ActionType] = DEFAULT_ACTION_TYPES,
        action_order: Literal[
            'simultaneous', 'round-robin', 'sequential', 'random'
        ] = 'sequential',
        evaluators: list[Evaluator] | None = None,
        model_name: str = 'gpt-4o-mini',
        terminal_evaluators: list[Evaluator] | None = None,
        background_class: type[TBackground] | None = None,
        max_turns: int = 20,
    ) -> None:
        """A chat environment for multiple agents.

        Args:
            available_action_types (set[ActionType], optional): The action types that are available to the agents. Defaults to set(["none", "speak", "non-verbal communication", "action"]).
            action_order (Literal["simultaneous", "round-robin", "sequential", "random"], optional): The order in which the agents take actions. Defaults to "sequential".
            model_name (str, optional): The name of the language model to use. Defaults to "gpt-4o-mini".
            max_turns (int, optional): Maximum number of turns before termination. Defaults to 20.
        """
        super().__init__()
        if background_class is None:
            self.background_class = MultiAgentChatBackground
        else:
            self.background_class = background_class
        self.background = self.background_class(
            scenario='',
            agent_configs=[],
        )

        self.agents = []
        self.agent_names = []
        self.action_spaces = {}
        self.available_action_types = list(available_action_types)
        self.action_order = action_order
        self.action_mask: list[bool] = []
        self.evaluators = evaluators or []
        self.terminal_evaluators = terminal_evaluators or []
        self.model_name = model_name
        self.turn_number = 0
        self.max_turns = max_turns
        self.inbox: list[tuple[str, Message]] = []
        self.current_agent_index = 0  # For sequential ordering

    def reset_inbox(self) -> None:
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: dict[str, Any] | None = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> dict[str, Observation]:
        """Reset the environment with new agents and background."""
        if seed is not None:
            random.seed(seed)

        self.turn_number = 0
        self.current_agent_index = 0
        self.reset_inbox()

        if agents is not None:
            self.agents = list(agents.values())
            self.agent_names = list(agents.keys())
        else:
            self.agents = []
            self.agent_names = []

        # Initialize action mask based on action order
        self._update_action_mask()

        # Create initial observations for all agents
        initial_obs = {}
        for agent_name in self.agent_names:
            agent_index = self.agent_names.index(agent_name)
            initial_obs[agent_name] = Observation(
                last_turn=self.background.to_natural_language(),
                turn_number=0,
                available_actions=(
                    list(self.available_action_types)
                    if agent_index < len(self.action_mask)
                    and self.action_mask[agent_index]
                    else ['none']
                ),
            )

        return initial_obs

    def _update_action_mask(self) -> None:
        """Update the action mask based on the current action order."""
        if not self.agent_names:
            self.action_mask = []
            return

        if self.action_order == 'simultaneous':
            self.action_mask = [True for _ in self.agent_names]
        elif self.action_order == 'round-robin':
            self.action_mask = [False for _ in self.agent_names]
            self.action_mask[self.turn_number % len(self.agent_names)] = True
        elif self.action_order == 'sequential':
            self.action_mask = [False for _ in self.agent_names]
            self.action_mask[self.current_agent_index] = True
        elif self.action_order == 'random':
            self.action_mask = [False for _ in self.agent_names]
            self.action_mask[random.randint(0, len(self.agent_names) - 1)] = True

    def get_turn_number(self) -> int:
        return self.turn_number

    def is_terminated(self) -> bool:
        return self.turn_number >= self.max_turns

    def get_observation(self, agent_name: str) -> Observation:
        """Get the current observation for a specific agent."""
        if agent_name not in self.agent_names:
            raise ValueError(f'Agent {agent_name} not found in environment')

        # Build the last turn description from recent actions
        last_turn = ''
        if self.inbox and self.turn_number > 0:
            recent_actions = []
            for source, message in self.inbox[-len(self.agent_names) :]:
                if isinstance(message, AgentAction) and message.action_type != 'none':
                    recent_actions.append(f'{source} {message.to_natural_language()}')

            if recent_actions:
                last_turn = '; '.join(recent_actions)
            else:
                last_turn = 'No recent actions'
        else:
            last_turn = self.background.to_natural_language()

        agent_index = self.agent_names.index(agent_name)
        available_actions = (
            list(self.available_action_types)
            if agent_index < len(self.action_mask) and self.action_mask[agent_index]
            else ['none']
        )

        return Observation(
            last_turn=last_turn,
            turn_number=self.turn_number,
            available_actions=available_actions,
        )

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    @validate_call
    def step(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        """Execute one step in the environment."""
        # Convert dict format to AgentAction format if needed
        if actions and isinstance(next(iter(actions.values())), dict):
            actions = {
                agent: AgentAction(
                    action_type=action_dict['action_type'],
                    argument=action_dict.get('argument', ''),
                )
                for agent, action_dict in actions.items()
            }

        # Record actions in inbox
        for agent_name, action in actions.items():
            self.recv_message(agent_name, action)

        # Update turn number and action mask
        self.turn_number += 1

        # Update current agent index for sequential ordering
        if self.action_order == 'sequential':
            self.current_agent_index = (self.current_agent_index + 1) % len(
                self.agent_names
            )

        self._update_action_mask()

        # Check for termination
        terminated = self.is_terminated()

        # Create observations for next turn
        observations = {}
        for agent_name in self.agent_names:
            observations[agent_name] = self.get_observation(agent_name)

        # Default rewards and info
        rewards = {agent_name: 0.0 for agent_name in self.agent_names}
        truncated = {agent_name: False for agent_name in self.agent_names}
        terminated_dict = {agent_name: terminated for agent_name in self.agent_names}
        info = {agent_name: {} for agent_name in self.agent_names}

        return observations, rewards, terminated_dict, truncated, info

    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        """Async version of step with evaluator support."""
        self.turn_number += 1

        complied_actions = self._process_actions(actions)

        self._record_turn_messages(complied_actions)

        response = await self._run_evaluators()

        self._update_state()

        observations = self._create_observations()
        rewards = {agent_name: 0.0 for agent_name in self.agent_names}
        truncated = {agent_name: False for agent_name in self.agent_names}
        terminated = self.is_terminated()
        terminated_dict = {agent_name: terminated for agent_name in self.agent_names}

        info = {agent_name: {} for agent_name in self.agent_names}
        if response:
            for agent_name in self.agent_names:
                info[agent_name] = {
                    'comments': response.comments or '',
                    'complete_rating': getattr(response, f'{agent_name}_rate', 0) or 0,
                }
            if response.terminated and self.terminal_evaluators:
                info['rewards_prompt'] = {
                    'overall_prompt': self.terminal_evaluators[0].prompt  # type: ignore
                }

        return observations, rewards, terminated_dict, truncated, info

    def _process_actions(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> dict[str, AgentAction]:
        """Process and convert actions to AgentAction format."""
        complied_actions: dict[str, AgentAction] = {}
        for key in actions.keys():
            action = actions[key]
            if isinstance(action, AgentAction):
                complied_actions[key] = action
            else:
                action['action_type'] = self.available_action_types[
                    int(action['action_type'])
                ]
                complied_actions[key] = AgentAction.parse_obj(action)

        for idx, agent in enumerate(self.agent_names):
            if not self.action_mask[idx]:
                complied_actions[agent] = AgentAction(action_type='none', argument='')

        return complied_actions

    def _record_turn_messages(self, complied_actions: dict[str, AgentAction]) -> None:
        """Record turn messages and actions."""
        self.recv_message(
            'Environment', SimpleMessage(message=f'Turn #{self.turn_number}')
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

    async def _run_evaluators(self) -> Any:
        """Run evaluators and return response."""
        response = None
        if self.evaluators:
            response = unweighted_aggregate_evaluate(
                list(
                    itertools.chain(
                        *await asyncio.gather(
                            *[
                                evaluator.__acall__(
                                    turn_number=self.turn_number,
                                    messages=self.inbox,
                                )
                                for evaluator in self.evaluators
                            ]
                        )
                    )
                )
            )

        # Run terminal evaluators if conversation is terminated
        if response and response.terminated and self.terminal_evaluators:
            terminal_response = unweighted_aggregate_evaluate(
                list(
                    itertools.chain(
                        *await asyncio.gather(
                            *[
                                evaluator.__acall__(
                                    turn_number=self.turn_number,
                                    messages=self.inbox,
                                )
                                for evaluator in self.terminal_evaluators
                            ]
                        )
                    )
                )
            )
            # incorporate terminal response into response
            if response.p1_rate is None:
                response.p1_rate = terminal_response.p1_rate
            if response.p2_rate is None:
                response.p2_rate = terminal_response.p2_rate
            if response.comments and terminal_response.comments:
                response.comments += terminal_response.comments
            elif terminal_response.comments:
                response.comments = terminal_response.comments

        return response

    def _update_state(self) -> None:
        """Update environment state."""
        # Update current agent index for sequential ordering
        if self.action_order == 'sequential':
            self.current_agent_index = (self.current_agent_index + 1) % len(
                self.agent_names
            )
        self._update_action_mask()

    def _create_observations(self) -> dict[str, Observation]:
        """Create observations for all agents."""
        observations = {}
        for agent_name in self.agent_names:
            observations[agent_name] = self.get_observation(agent_name)
        return observations

    async def evaluate_episode(self) -> dict:
        """Evaluate the episode using terminal evaluators."""
        if not self.terminal_evaluators:
            return {'message': 'No terminal evaluators available'}

        try:
            response = unweighted_aggregate_evaluate(
                list(
                    itertools.chain(
                        *await asyncio.gather(
                            *[
                                evaluator.__acall__(
                                    turn_number=self.turn_number,
                                    messages=self.inbox,
                                )
                                for evaluator in self.terminal_evaluators
                            ]
                        )
                    )
                )
            )

            result = {
                'terminated': response.terminated,
                'comments': response.comments or '',
            }

            # Add scores for each agent
            for agent_name in self.agent_names:
                agent_rate = getattr(response, f'{agent_name}_rate', None)
                if agent_rate is not None:
                    result[f'{agent_name}_score'] = agent_rate
                    if hasattr(response, f'{agent_name}_details'):
                        result[f'{agent_name}_details'] = getattr(
                            response, f'{agent_name}_details'
                        )

            return result
        except Exception as e:
            return {'message': f'Evaluation failed: {str(e)}'}

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        summary = 'Multi-Agent Conversation Summary\n'
        summary += f'Total turns: {self.turn_number}\n'
        summary += f"Agents: {', '.join(self.agent_names)}\n"
        summary += f'Action order: {self.action_order}\n\n'

        if self.inbox:
            summary += 'Conversation history:\n'
            for source, message in self.inbox:
                if (
                    source != 'Environment'
                    or not message.to_natural_language().startswith('Turn #')
                ):
                    summary += f'{source}: {message.to_natural_language()}\n'

        return summary

    def render(self, mode: str = 'human') -> None:
        """Render the current state of the environment."""
        print(f'Turn {self.turn_number}')
        print(
            f'Active agents: {[name for i, name in enumerate(self.agent_names) if self.action_mask[i]]}'
        )
        if self.inbox:
            print('Recent messages:')
            for source, message in self.inbox[-5:]:  # Show last 5 messages
                print(f'  {source}: {message.to_natural_language()}')

    def close(self) -> None:
        """Clean up resources."""
        pass
