import asyncio
import itertools
import random
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

from pydantic import validate_call

from tiny_chat.evaluator import Evaluator, unweighted_aggregate_evaluate
from tiny_chat.messages import (
    ActionType,
    AgentAction,
    Message,
    Observation,
    ScriptBackground,
    SimpleMessage,
    TinyChatBackground,
)
from tiny_chat.utils.logger import logger

TBackground = TypeVar('TBackground', bound=ScriptBackground)

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
        if action.action_type != 'none':
            if action_str != '':
                action_str += ';'
            action_str += f'{agent} {action.to_natural_language()}'
    return action_str


class BaseChatEnivronment(ABC):
    @abstractmethod
    def reset(self, **kwargs: Any) -> Any:
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


class TinyChatEnvironment(BaseChatEnivronment):
    def __init__(
        self,
        available_action_types: set[ActionType] = DEFAULT_ACTION_TYPES,
        action_order: Literal[
            'simultaneous', 'round-robin', 'sequential', 'random', 'agent_id_based'
        ] = 'simultaneous',
        evaluators: list[Evaluator] | None = None,
        model_name: str = 'gpt-4o-mini',
        terminal_evaluators: list[Evaluator] | None = None,
        max_turns: int = 20,
        speaking_order: list[int] | None = None,
    ) -> None:
        """Initialize the unified chat environment."""
        super().__init__()
        self.available_action_types = list(available_action_types)
        self.action_order = action_order
        self.evaluators = evaluators or []
        self.terminal_evaluators = terminal_evaluators or []
        self.model_name = model_name
        self.max_turns = max_turns
        self.speaking_order = speaking_order

        self.agents: list[Any] = []
        self.agent_names: list[str] = []
        self.background: TinyChatBackground | None = None
        self.env_background: TinyChatBackground | None = None
        self.turn_number = 0
        self.current_agent_index = 0
        self.action_mask: list[bool] = []
        self.inbox: list[tuple[str, Message]] = []

        self.action_spaces: dict[str, Any] = {}
        self._omniscient: bool = False

    def reset_inbox(self) -> None:
        """Reset the message inbox."""
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        """Receive a message and add to inbox."""
        self.inbox.append((source, message))

    def reset(  # type: ignore[override]
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

        if agents is None or not agents:
            raise ValueError('agents must be provided')

        self.agents = list(agents.values())
        self.agent_names = list(agents.keys())
        self._omniscient = bool(omniscient)

        if self.action_order == 'agent_id_based':
            self._sort_agents_by_speaking_order(agents)

        self._setup_unified_background(agents, options, omniscient, lite)

        self.action_spaces = {
            agent: {
                'action_type': list(range(len(self.available_action_types))),
                'argument': str,
            }
            for agent in self.agent_names
        }

        self._update_action_mask()

        self.recv_message('Environment', self.env_background or self.background)  # type: ignore

        initial_obs = {}
        for i, agent_name in enumerate(self.agent_names):
            if omniscient:
                bg_for_agent = self.env_background
            else:
                bg_for_agent = (
                    self.env_background or self.background
                ).create_agent_specific_background(  # type: ignore
                    agent_name, omniscient=False
                )
            initial_obs[agent_name] = Observation(
                last_turn=bg_for_agent.to_natural_language() if bg_for_agent else '',
                turn_number=0,
                available_actions=(
                    list(self.available_action_types)
                    if i < len(self.action_mask) and self.action_mask[i]
                    else ['none']
                ),
            )

        return initial_obs

    def _sort_agents_by_speaking_order(self, agents: dict[str, Any]) -> None:
        """Sort agents by their speaking_order list."""
        if not self.speaking_order:
            return

        agent_items = list(agents.items())

        temp_id_counter = len(self.speaking_order)
        for agent_name, agent in agent_items:
            if not hasattr(agent, 'speaking_id'):
                agent.speaking_id = temp_id_counter
                temp_id_counter += 1

        def get_speaking_order(item):
            agent_name, agent = item
            if hasattr(agent, 'speaking_id'):
                agent_id = agent.speaking_id
                try:
                    return self.speaking_order.index(agent_id)
                except ValueError:
                    return len(self.speaking_order)
            return len(self.speaking_order)

        agent_items.sort(key=get_speaking_order)

        sorted_agents = dict(agent_items)
        self.agents = list(sorted_agents.values())
        self.agent_names = list(sorted_agents.keys())

        agents.clear()
        agents.update(sorted_agents)

    def _setup_unified_background(
        self,
        agents: dict[str, Any],
        options: dict[str, str] | None,
        omniscient: bool,
        lite: bool,
    ) -> None:
        """Setup unified background for any number of agents."""
        agent_configs = []

        for agent_name, agent in agents.items():
            config = {'name': agent_name}

            if hasattr(agent, 'speaking_id'):
                agent_id = agent.speaking_id
            else:
                agent_id = len(self.speaking_order) if self.speaking_order else 0

            if not lite:
                if hasattr(agent, 'profile'):
                    profile = agent.profile
                    if hasattr(profile, 'to_background_string'):
                        config['background'] = profile.to_background_string(agent_id)
                    elif hasattr(profile, 'description'):
                        config['background'] = (
                            f"<root><p viewer='agent_{agent_id}'>{profile.description}</p></root>"
                        )
                elif hasattr(agent, 'description'):
                    config['background'] = (
                        f"<root><p viewer='agent_{agent_id}'>{agent.description}</p></root>"
                    )
                else:
                    config['background'] = (
                        f"<root><p viewer='agent_{agent_id}'>Agent {agent_id}</p></root>"
                    )
            else:
                config['background'] = ''

            if hasattr(agent, 'goal'):
                config['goal'] = f"<root viewer='agent_{agent_id}'>{agent.goal}</root>"
            else:
                config['goal'] = (
                    f"<root viewer='agent_{agent_id}'>Achieve your objectives in this conversation</root>"
                )

            agent_configs.append(config)

        num_agents = len(agents)
        if num_agents == 2:
            scenario = (
                options.get('scenario', 'A conversation between two agents')
                if options
                else 'A conversation between two agents'
            )
        else:
            scenario = (
                options.get('scenario', f'A conversation between {num_agents} agents')
                if options
                else f'A conversation between {num_agents} agents'
            )

        self.env_background = TinyChatBackground(
            scenario=scenario,
            agent_configs=agent_configs,
        )
        self.background = self.env_background

    def _update_action_mask(self) -> None:
        """Update the action mask based on the current action order."""
        if not self.agent_names:
            self.action_mask = []
            return

        if self.action_order == 'simultaneous':
            self.action_mask = [True for _ in self.agent_names]
        elif self.action_order == 'round-robin':
            self.action_mask = [False for _ in self.agent_names]
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == 'sequential':
            self.action_mask = [False for _ in self.agent_names]
            self.action_mask[self.current_agent_index] = True
        elif self.action_order == 'random':
            self.action_mask = [False for _ in self.agent_names]
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        elif self.action_order == 'agent_id_based':
            self.action_mask = [True for _ in self.agent_names]

    def get_turn_number(self) -> int:
        """Get the current turn number."""
        return self.turn_number

    def is_terminated(self) -> bool:
        """Check if the conversation is terminated."""
        return self.turn_number >= self.max_turns

    def get_observation(self, agent_name: str) -> Observation:
        """Get the current observation for a specific agent."""
        if agent_name not in self.agent_names:
            raise ValueError(f'Agent {agent_name} not found in environment')

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
            if self._omniscient:
                last_turn = (
                    self.env_background or self.background
                ).to_natural_language()  # type: ignore
            else:
                bg_for_agent = (
                    self.env_background or self.background
                ).create_agent_specific_background(  # type: ignore
                    agent_name, omniscient=False
                )
                last_turn = bg_for_agent.to_natural_language()

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
        """Get the number of agents."""
        return len(self.agent_names)

    @validate_call
    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        """Execute one step in the environment (asynchronous version with evaluators)."""
        self.turn_number += 1

        complied_actions = self._process_actions(actions)
        self._mask_actions(complied_actions)
        self._record_turn_messages(complied_actions)

        response = await self._run_evaluators()

        self._update_state()

        obs = _actions_to_natural_language(complied_actions)
        observations = self._create_observations(obs)

        rewards = self._build_rewards(response)
        truncated = dict.fromkeys(self.agent_names, False)
        terminated_flag = self.is_terminated() or bool(
            getattr(response, 'terminated', False)
        )
        terminated_dict = dict.fromkeys(self.agent_names, terminated_flag)
        info = self._build_info(response)

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

        for agent_name in self.agent_names:
            if agent_name not in complied_actions:
                complied_actions[agent_name] = AgentAction(
                    action_type='none', argument=''
                )
        return complied_actions

    def _mask_actions(self, actions: dict[str, AgentAction]) -> None:
        """Mask actions for agents that are not in turn."""
        for idx, agent in enumerate(self.agent_names):
            if not self.action_mask[idx]:
                actions[agent] = AgentAction(action_type='none', argument='')

    def _record_turn_messages(self, complied_actions: dict[str, AgentAction]) -> None:
        """Record turn messages and actions."""
        self.recv_message(
            'Environment', SimpleMessage(message=f'Turn #{self.turn_number}')
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

    def _update_state(self) -> None:
        """Update environment state."""
        if self.action_order == 'sequential':
            self.current_agent_index = (self.current_agent_index + 1) % len(
                self.agent_names
            )
        self._update_action_mask()

    def _create_observations(self, obs: str) -> dict[str, Observation]:
        """Create observations for all agents."""
        observations = {}
        for i, agent_name in enumerate(self.agent_names):
            observations[agent_name] = Observation(
                last_turn=obs,
                turn_number=self.turn_number,
                available_actions=(
                    list(self.available_action_types)
                    if i < len(self.action_mask) and self.action_mask[i]
                    else ['none']
                ),
            )
        return observations

    async def _run_evaluators(self) -> Any:
        """Run evaluators and return response."""
        if not self.evaluators:
            return None

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
            self._merge_terminal_response(response, terminal_response)

        return response

    def _merge_terminal_response(self, response: Any, terminal_response: Any) -> None:
        """Helper method to merge terminal response into main response."""
        if (
            hasattr(terminal_response, 'per_agent_scores')
            and terminal_response.per_agent_scores
        ):
            if not hasattr(response, 'per_agent_scores'):
                response.per_agent_scores = {}
            for agent_name, agent_score in terminal_response.per_agent_scores.items():
                if agent_name not in response.per_agent_scores:
                    response.per_agent_scores[agent_name] = agent_score

        if response.comments and terminal_response.comments:
            response.comments += terminal_response.comments
        elif terminal_response.comments:
            response.comments = terminal_response.comments

    def _build_rewards(self, response: Any) -> dict[str, float]:
        """Build rewards dictionary from evaluator response."""
        rewards = dict.fromkeys(self.agent_names, 0.0)

        if (
            response
            and hasattr(response, 'per_agent_scores')
            and response.per_agent_scores
        ):
            for agent_name in self.agent_names:
                if agent_name in response.per_agent_scores:
                    agent_score = response.per_agent_scores[agent_name]
                    # Extract the main score from tuple format (score, details) or direct float
                    if isinstance(agent_score, tuple) and len(agent_score) >= 1:
                        rewards[agent_name] = float(agent_score[0]) or 0.0
                    elif isinstance(agent_score, int | float):
                        rewards[agent_name] = float(agent_score) or 0.0

        return rewards

    def _build_info(self, response: Any) -> dict[str, Any]:
        """Build info dictionary from evaluator response."""
        info: dict[str, Any] = dict.fromkeys(self.agent_names, {})

        if response:
            for agent_name in self.agent_names:
                agent_score = 0.0
                agent_details = {}

                if hasattr(response, 'per_agent_scores') and response.per_agent_scores:
                    if agent_name in response.per_agent_scores:
                        score_data = response.per_agent_scores[agent_name]
                        if isinstance(score_data, tuple) and len(score_data) >= 2:
                            agent_score = score_data[0]
                            agent_details = (
                                score_data[1] if isinstance(score_data[1], dict) else {}
                            )
                        elif isinstance(score_data, int | float):
                            agent_score = score_data

                info[agent_name] = {
                    'comments': response.comments or '',
                    'complete_rating': agent_score,
                    'detailed_scores': agent_details,
                }

            if response.terminated and self.terminal_evaluators:
                info['rewards_prompt'] = {
                    'overall_prompt': self.terminal_evaluators[0].prompt  # type: ignore
                }

        return info

    async def evaluate_episode(self) -> dict[str, Any]:
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

            if hasattr(response, 'per_agent_scores') and response.per_agent_scores:
                for agent_name in self.agent_names:
                    if agent_name in response.per_agent_scores:
                        score_data = response.per_agent_scores[agent_name]
                        if isinstance(score_data, tuple) and len(score_data) >= 2:
                            result[f'{agent_name}_score'] = score_data[0]
                            if isinstance(score_data[1], dict):
                                result[f'{agent_name}_details'] = score_data[1]
                        elif isinstance(score_data, int | float):
                            result[f'{agent_name}_score'] = score_data

            return result
        except Exception as e:
            return {'message': f'Evaluation failed: {str(e)}'}

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        summary = 'TinyChat Conversation Summary\n'
        summary += f'Total turns: {self.turn_number}\n'
        summary += f"Agents: {', '.join(self.agent_names)}\n"
        summary += f'Action order: {self.action_order}\n\n'

        if self.inbox:
            summary += 'Conversation history:\n'
            for source, message in self.inbox:
                if source == 'Environment':
                    continue
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
            for source, message in self.inbox[-5:]:
                print(f'  {source}: {message.to_natural_language()}')

    def close(self) -> None:
        """Clean up resources."""
        pass
