import re
from typing import Literal, cast

from pydantic import BaseModel, Field

from tiny_chat.utils import format_docstring

ActionType = Literal['none', 'speak', 'non-verbal communication', 'action', 'leave']


class Message(BaseModel):
    """
    An interface for messages.
    There is only one required method: to_natural_language
    """

    def to_natural_language(self) -> str:
        raise NotImplementedError


class SimpleMessage(Message):
    """
    A simple message with a single string field.
    """

    message: str = Field(description='the message')

    def to_natural_language(self) -> str:
        return self.message


class MessengerMixin:
    def __init__(self) -> None:
        self.inbox: list[tuple[str, Message]] = []

    def reset_inbox(self) -> None:
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))


class Observation(Message):
    last_turn: str = Field(description='the last turn of the conversation')
    turn_number: int = Field(description='the turn number of the conversation')
    available_actions: list[ActionType] = Field(description='the available actions')

    def to_natural_language(self) -> str:
        if self.turn_number == 0:
            return f'\n{self.last_turn}\nConversation Starts:\n'
        else:
            return f'Turn #{self.turn_number - 1}: {self.last_turn}\n'


class ScriptBackground(Message):
    def to_natural_language(self) -> str:
        raise NotImplementedError


class UnifiedChatBackground(ScriptBackground):
    """A unified background class that handles both 2-agent and multi-agent scenarios."""

    scenario: str = Field(description='scenario of the episode')
    agent_configs: list[dict] = Field(description='configurations of all agents')

    def to_natural_language(self) -> str:
        """Generate natural language description of the background."""
        agent_info = ''
        for i, config in enumerate(self.agent_configs):
            agent_name = config.get('name', f'Agent{i+1}')
            agent_info += f'\n{agent_name}:'
            if 'background' in config and config['background']:
                agent_info += f"\n  Background: {config['background']}"
            if 'goal' in config:
                agent_info += f"\n  Goal: {config['goal']}"

        return format_docstring(
            f"""Here is the context of this interaction:
            Scenario: {self.scenario}
            Participants: {agent_info}
            """
        )

    def get_agent_goal(self, agent_name: str) -> str:
        """Get the goal for a specific agent."""
        for config in self.agent_configs:
            if config.get('name') == agent_name and 'goal' in config:
                return config['goal']
        return 'Achieve your objectives in this conversation'

    def get_agent_background(self, agent_name: str) -> str:
        """Get the background for a specific agent."""
        for config in self.agent_configs:
            if config.get('name') == agent_name and 'background' in config:
                return config['background']
        return ''

    def create_agent_specific_background(
        self, target_agent_name: str, omniscient: bool = False
    ) -> 'UnifiedChatBackground':
        """Create a background specific to one agent, optionally hiding other agents' goals."""
        agent_configs = []

        for config in self.agent_configs:
            agent_name = config.get('name', '')
            new_config = config.copy()

            if (
                not omniscient
                and agent_name != target_agent_name
                and 'goal' in new_config
            ):
                new_config['goal'] = 'Unknown'

            agent_configs.append(new_config)

        return UnifiedChatBackground(
            scenario=self.scenario, agent_configs=agent_configs
        )


class ScriptEnvironmentResponse(Message):
    terminated: bool = Field(
        description='whether the conversation is terminated',
        default=False,
    )
    p1_rate: float | tuple[float, dict[str, float]] | None = Field(
        description='rating of participant 1, on the scale of 1 to 10',
        default=None,
    )
    p2_rate: float | tuple[float, dict[str, float]] | None = Field(
        description='rating of participant 2, on the scale of 1 to 10',
        default=None,
    )
    per_agent_scores: dict[str, float | tuple[float, dict[str, float]]] = Field(
        description="ratings for arbitrary number of agents, keyed by agent name or agent index label (e.g., 'agent_1')",
        default_factory=dict,
    )
    comments: str | None = Field(
        description='All of the comments supporting the termination and rating',
        default=None,
    )

    def to_natural_language(self) -> str:
        reason_to_stop = format_docstring(
            f"""Environment response:
        {'The conversation is terminated.' if self.terminated else ''}
        {'Rating of participant 1' + str(self.p1_rate) if self.p1_rate is not None else ''}
        {'Rating of participant 2' + str(self.p2_rate) if self.p2_rate is not None else ''}
        {self.comments if self.comments is not None else ''}
        """
        )
        clean_text = ''
        for line in reason_to_stop.split('\n'):
            if line.strip():
                clean_text += line + '\n'
        return clean_text


class AgentAction(Message):
    action_type: ActionType = Field(
        description='whether to speak at this turn or choose to not do anything'
    )
    argument: str = Field(
        description='the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action'
    )

    def to_natural_language(self) -> str:
        match self.action_type:
            case 'none':
                return 'did nothing'
            case 'speak':
                return f'said: "{self.argument}"'
            case 'non-verbal communication':
                return f'[{self.action_type}] {self.argument}'
            case 'action':
                return f'[{self.action_type}] {self.argument}'
            case 'leave':
                return 'left the conversation'


ScriptInteractionReturnType = tuple[
    list[list[tuple[str, str, Message]]], list[tuple[str, Message]]
]


class ScriptInteraction(Message):
    interactions: str = Field(
        description="""The interaction between the two participants in maximum 20 turns. Each turn is separated by a newline, and should only describe one agent. Following the structure:
        Turn #x
        [participant's name] [action] {argument for some actions}

        You can use different types of actions, but only use one in each turn. You should move other information into argument part. Below shows a python code snippet of the format for each action type:
        match self.action_type:
            case "none":
                return "did nothing"
            case "speak":
                return f'said: "{self.argument}"'
            case "non-verbal communication":
                return f"[{self.action_type}] {self.argument}"
            case "action":
                return f"[{self.action_type}] {self.argument}"
            case "leave":
                return "left the conversation"

        For example, the following is acceptable:
        Turn #x
        Oliver Thompson said: "Hey Esmeralda, what's wrong? You seem upset."
        Turn #x
        Esmeralda Solis [action] moved closer
        Turn #x
        Oliver Thompson [non-verbal communication] smiled
        Turn #x
        Esmeralda Solis did nothing
        Turn #x
        Oliver Thompson left the conversation
        Turn #x
        Esmeralda Solis [action] leaned in and lowered her voice: "Sorry"

        And the following is not acceptable:
        Turn #1
        Oliver Thompson [speak] said: "Hey Esmeralda, what's wrong? You seem upset."
        Turn #1
        Esmeralda Solis non-verbal communication moved closer
        """
    )

    def to_natural_language(self) -> str:
        return self.interactions

    def parse(
        self, agent_names: list[str], background: str
    ) -> ScriptInteractionReturnType:
        interaction = self.interactions
        # print("Interaction: ", interaction)
        lines = self.split_by_turn(interaction)

        agent_results: list[tuple[str, Message]] = []
        results: list[list[tuple[str, str, Message]]] = [
            [
                (
                    'Environment',
                    name,
                    Observation(
                        last_turn=background,
                        turn_number=0,
                        available_actions=['none'],
                    ),
                )
                for name in agent_names
            ]
        ]

        for line_idx, line in enumerate(lines):
            try:
                res = self.parse_single_dialogue(line)
                action: ActionType = cast(ActionType, res['action'])
                argument: str = cast(str, res['argument'])
                cast(int, res['turn'])
                name: str = cast(str, res['name'])

                parsed_action = AgentAction(action_type=action, argument=argument)
                if name not in agent_names:
                    print(
                        f'The name of the agent, {name}, is not in the list of agent names, {agent_names}'
                    )
                    name = agent_names[line_idx % 2]
            except Exception as e:
                print(
                    f'Error when parsing the dialogue: {line}',
                    f'The error is: {e}',
                )
                raise e
            parsed_action = AgentAction(action_type='none', argument='')
            name = agent_names[line_idx % 2]  # TODO same question as above
            inactive_agent_name = (
                agent_names[0] if name == agent_names[1] else agent_names[1]
            )
            results.append(
                [
                    (
                        'Environment',
                        name,
                        Observation(
                            last_turn='environment is the agent',
                            turn_number=line_idx + 1,
                            available_actions=['none'],
                        ),
                    )
                    for name in agent_names
                ]
                + [
                    (name, 'Environment', parsed_action),
                    (
                        inactive_agent_name,
                        'Environment',
                        AgentAction(action_type='none', argument='did nothing'),
                    ),
                ]
            )

            agent_results.append((name, parsed_action))
        # print("Parsed agent results: ", agent_results)
        return (results, agent_results)

    def parse_single_dialogue(
        self, dialogue: str
    ) -> dict[str, str | int | ActionType | None]:
        """Parse a single dialogue string and return a dictionary with turn, name, action, and argument."""

        match_turn_name = re.match(
            r"Turn #?(\d+):?\s*\n((?:[A-Z]['a-z]* ?)+)", dialogue
        )

        if not match_turn_name:
            raise ValueError(
                f'The dialogue does not match the expected format: {dialogue}'
            )

        turn, name = match_turn_name.groups()
        action_content = dialogue[len(match_turn_name.group(0)) :].strip()

        if 'did nothing' in action_content:
            action, argument = 'none', ''
        elif match := re.match(r'said: "(.*?)"', action_content):
            action, argument = 'speak', match.group(1)
            action, argument = action.strip(), argument.strip()
        elif match := re.match(r'\[speak\] said: "(.*?)"', action_content):
            action, argument = 'speak', match.group(1)
            action, argument = action.strip(), argument.strip()
        elif match := re.match(
            r'\[(non-verbal communication|action)\] (.*)', action_content
        ):
            action, argument = match.groups()
        elif 'left the conversation' in action_content:
            action, argument = 'leave', ''
        else:
            action, argument = None, None

        parsed_item = {
            'turn': int(turn),
            'name': name.strip(),
            'action': action,
            'argument': argument,
        }
        return parsed_item

    def split_by_turn(self, input_string: str) -> list[str]:
        """Split the input dialogue string by turn and return a list of dialogues."""
        dialogues = re.split(r'(?=Turn #?\d+)', input_string)
        dialogues = [dialogue.strip() for dialogue in dialogues if dialogue.strip()]
        dialogues = [dialogue for dialogue in dialogues if dialogue.startswith('Turn')]
        dialogues[-1] = '\n'.join(dialogues[-1].split('\n')[:2])

        for dialogue in dialogues:
            # TODO this is current workaround for the issue of multiple agents in one turn
            if len(dialogue.split('\n')) >= 3:
                raise ValueError('Only one agent can act per turn.')
        return dialogues

    @staticmethod
    def default_value_for_return_type() -> ScriptInteractionReturnType:
        results_1: list[list[tuple[str, str, Message]]] = [
            [
                (
                    'Environment',
                    name,
                    Observation(
                        last_turn='Environment is the agent',
                        turn_number=0,
                        available_actions=['none'],
                    ),
                )
                for name in ['none', 'none']
            ]
        ]
        results_2: list[tuple[str, Message]] = [
            ('', AgentAction(action_type='none', argument=''))
        ]
        return (results_1, results_2)
