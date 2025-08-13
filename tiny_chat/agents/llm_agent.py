from collections.abc import Iterable
from typing import cast

from tiny_chat.generator import agenerate_action, agenerate_goal
from tiny_chat.messages import AgentAction, Observation
from tiny_chat.profile import BaseAgentProfile

from .base_agent import BaseAgent


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: BaseAgentProfile | dict | None = None,
        profile_jsonl_path: str | None = None,
        model_name: str = 'gpt-4o-mini',
        script_like: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            profile_jsonl_path=profile_jsonl_path,
        )
        self.model_name = model_name
        self.script_like = script_like

    async def act(self, obs: Observation) -> AgentAction:
        self.recv_message('Environment', obs)

        await self._ensure_goal()

        if self._only_none_action(obs.available_actions):
            return AgentAction(action_type='none', argument='')

        action = await agenerate_action(
            self.model_name,
            history=self._history_text(self.inbox),
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=self.goal,
            script_like=self.script_like,
        )
        return cast(AgentAction, action)

    async def _ensure_goal(self) -> None:
        if self._goal is not None:
            return
        background = self._first_message_text() or ''
        self._goal = await agenerate_goal(self.model_name, background=background)

    def _first_message_text(self) -> str | None:
        if not getattr(self, 'inbox', None):
            return None
        return self.inbox[0][1].to_natural_language()

    @staticmethod
    def _history_text(inbox: Iterable[tuple[str, object]]) -> str:
        return '\n'.join(msg.to_natural_language() for _, msg in inbox)

    @staticmethod
    def _only_none_action(actions: Iterable[str]) -> bool:
        acts = list(actions)
        return len(acts) == 1 and acts[0].casefold() == 'none'
