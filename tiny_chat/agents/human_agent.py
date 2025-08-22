import asyncio
from collections.abc import Iterable
from typing import Any

from tiny_chat.messages import AgentAction, Observation
from tiny_chat.profiles import BaseAgentProfile

from .base_agent import BaseAgent

HELP_TEXT = """\
[HumanAgent Command Help]
- Press Enter directly: action_type=none
- Enter one or more lines of text: default action_type=speak
- Explicitly specify action: write `action_type: <none|speak|non-verbal communication|action|leave>`
  on the first line, then provide the content on the following lines.
  End with an empty line.
Example:
  action_type: non-verbal communication
  smiling and nodding

  (empty line ends input)
"""


class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human-controlled agent that interacts via the terminal.
    Input is collected from stdin, supporting multi-line entries.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: BaseAgentProfile | dict[str, Any] | None = None,
        profile_jsonl_path: str | None = None,
        show_help: bool = True,
    ) -> None:
        if isinstance(agent_profile, dict):
            agent_profile = BaseAgentProfile(**agent_profile)

        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            profile_jsonl_path=profile_jsonl_path,
        )
        self.show_help = show_help

    async def act(self, obs: Observation) -> AgentAction:
        """
        Prompt the human user for input based on the current Observation.
        Returns an AgentAction according to the entered command or text.
        """
        self.recv_message('Environment', obs)

        print(f'\n========== Human Agent: {self.agent_name} ==========')
        if self.show_help:
            print(HELP_TEXT)
        print(
            f'[Turn {obs.turn_number}] Available actions: {list(obs.available_actions)}'
        )
        print('—— Context ——')
        print(obs.to_natural_language().rstrip())
        print('—— Enter your input (empty line to finish) ——')

        lines = await self._read_multiline()

        if not any(line.strip() for line in lines):
            return AgentAction(action_type='none', argument='')

        action_type, arg_lines = self._parse_action(lines, default_action='speak')

        if action_type not in obs.available_actions:
            print(f"[Note] Action '{action_type}' not allowed, replaced with 'none'")
            return AgentAction(action_type='none', argument='')

        argument = '\n'.join(arg_lines).strip()
        if action_type in ('leave', 'none'):
            argument = ''

        return AgentAction(action_type=action_type, argument=argument)

    async def _read_multiline(self) -> list[str]:
        """
        Read multiple lines of input from stdin.
        An empty line indicates the end of input.
        """
        loop = asyncio.get_running_loop()
        lines: list[str] = []

        async def _readline() -> str:
            return await loop.run_in_executor(None, input)

        while True:
            line = await _readline()
            if line.strip() == '':
                break
            lines.append(line)
        return lines

    @staticmethod
    def _parse_action(
        lines: Iterable[str], default_action: str = 'speak'
    ) -> tuple[str, list[str]]:
        """
        Check if the first line specifies an action type.
        Otherwise, default to `default_action`.
        """
        lines = list(lines)
        action = default_action
        content_start = 0

        if lines:
            first = lines[0].strip()
            prefix = 'action_type:'
            if first.lower().startswith(prefix):
                cand = first[len(prefix) :].strip().lower()
                if cand in {
                    'none',
                    'speak',
                    'non-verbal communication',
                    'action',
                    'leave',
                }:
                    action = cand
                    content_start = 1

        return action, lines[content_start:]
