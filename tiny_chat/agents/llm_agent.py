from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from tiny_chat.generator import agenerate_action, agenerate_goal
from tiny_chat.messages import AgentAction, Observation
from tiny_chat.profiles import BaseAgentProfile

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from tiny_chat.server.providers.base import BaseModelProvider


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: BaseAgentProfile | dict[str, Any] | None = None,
        profile_jsonl_path: str | None = None,
        model_name: str = 'gpt-4o-mini',
        script_like: bool = False,
        model_provider: 'BaseModelProvider | None' = None,
    ) -> None:
        if isinstance(agent_profile, dict):
            agent_profile = BaseAgentProfile(**agent_profile)

        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            profile_jsonl_path=profile_jsonl_path,
        )
        self.model_name = model_name
        self.script_like = script_like
        self._model_provider = model_provider

    async def act(self, obs: Observation) -> AgentAction:
        self.recv_message('Environment', obs)
        await self._ensure_goal()

        if self._only_none_action(obs.available_actions):
            return AgentAction(action_type='none', argument='')

        if self._model_provider:
            return await self._model_provider.agenerate_action(
                history=self._history_text(self.inbox),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
            )

        return await agenerate_action(
            self.model_name,
            history=self._history_text(self.inbox),
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=self.goal,
            script_like=self.script_like,
        )

    async def _ensure_goal(self) -> None:
        if self._goal is not None:
            return
        background = self._first_message_text() or ''

        if self._model_provider:
            self._goal = await self._model_provider.agenerate_goal(
                background=background
            )
        else:
            self._goal = await agenerate_goal(self.model_name, background=background)

    @property
    def uses_provider(self) -> bool:
        """是否使用 ModelProvider"""
        return self._model_provider is not None

    @property
    def effective_model_name(self) -> str:
        """有效的模型名称"""
        if self._model_provider:
            return self._model_provider._get_agenerate_model_name()
        return self.model_name

    @property
    def provider_type(self) -> str:
        """Provider 类型描述"""
        if self._model_provider:
            return f'{self._model_provider.__class__.__name__}({self._model_provider.type})'
        return 'direct_agenerate'

    def set_model_provider(self, provider: 'BaseModelProvider') -> None:
        """设置 ModelProvider"""
        self._model_provider = provider

    def _first_message_text(self) -> str | None:
        if not getattr(self, 'inbox', None):
            return None
        return self.inbox[0][1].to_natural_language()

    @staticmethod
    def _history_text(inbox: Iterable[tuple[str, Any]]) -> str:
        return '\n'.join(msg.to_natural_language() for _, msg in inbox)

    @staticmethod
    def _only_none_action(actions: Iterable[str]) -> bool:
        acts = list(actions)
        return len(acts) == 1 and acts[0].casefold() == 'none'
