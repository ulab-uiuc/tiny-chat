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
    Observation,
    ScriptBackground,
    SimpleMessage,
)

TBackground = TypeVar('TBackground', bound=ScriptBackground)


def _actions_to_natural_language(actions: dict[str, AgentAction]) -> str:
    action_str = ''
    for agent, action in actions.items():
        # Only record actions that did something
        if action.action_type != 'none':
            if action_str != '':
                action_str += ';'  # separate actions with semicolon
            action_str += f'{agent} {action.to_natural_language()}'
    return action_str


class TinyChatEnvironment:
    def __init__(
        self,
        available_action_types: set[ActionType] = set(
            ['none', 'speak', 'non-verbal communication', 'action', 'leave']
        ),
        action_order: Literal['simultaneous', 'round-robin', 'random'] = 'simultaneous',
        evaluators: list[Evaluator] = [],
        model_name: str = 'gpt-4o-mini',
        terminal_evaluators: list[Evaluator] = [],
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
        self.evaluators = evaluators
        self.terminal_evaluators = terminal_evaluators
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
            assert len(agents) == 2, 'Only supporting two agents right now'
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
            for i in range(self.num_agents):
                agent_backgrounds.append(copy.deepcopy(self.background))
        else:
            for i in range(self.num_agents):
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
        # 获取上一轮对话内容
        last_turn = ""
        if self.turn_number > 0 and self.inbox:
            # 从收件箱中获取上一轮的动作信息
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
        
        # 根据智能体名称确定动作掩码索引
        agent_index = 0  # 默认索引
        if hasattr(self, 'agents') and self.agents:
            try:
                agent_index = self.agents.index(agent_name)
            except ValueError:
                agent_index = 0
        
        # 确定可用的动作
        available_actions = ['none']  # 默认只有 'none' 动作
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

        if response.terminated:
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
            response.p1_rate = response.p1_rate or terminal_response.p1_rate
            response.p2_rate = response.p2_rate or terminal_response.p2_rate
            if response.comments and terminal_response.comments:
                response.comments += terminal_response.comments
            elif terminal_response.comments:
                response.comments = terminal_response.comments

        self.action_mask = [False for _ in self.agents]
        if self.action_order == 'round-robin':
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == 'random':
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]
        obs = _actions_to_natural_language(complied_actions)
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
            info,
        )

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        pass
