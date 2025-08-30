from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from tiny_chat.messages import AgentAction, Observation
from tiny_chat.profiles import BaseAgentProfile

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from tiny_chat.providers import BaseModelProvider


def _create_default_model_provider() -> 'BaseModelProvider':
    """Create a default model provider using gpt-4o-mini"""
    from tiny_chat.config import ModelProviderConfig
    from tiny_chat.providers import ModelProviderFactory

    default_config = ModelProviderConfig(
        name='gpt-4o-mini', type='openai', temperature=0.7
    )
    return ModelProviderFactory.create_provider(default_config)


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: BaseAgentProfile | dict[str, Any] | None = None,
        profile_jsonl_path: str | None = None,
        model_provider: 'BaseModelProvider | None' = None,
        script_like: bool = False,
    ) -> None:
        if isinstance(agent_profile, dict):
            agent_profile = BaseAgentProfile(**agent_profile)

        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            profile_jsonl_path=profile_jsonl_path,
        )

        self._model_provider = model_provider or _create_default_model_provider()
        self.script_like = script_like

    @property
    def speaking_id(self) -> int:
        """Get the speaking ID of the agent"""
        if hasattr(self, 'profile') and self.profile:
            return getattr(self.profile, 'speaking_id', 0)
        return 0

    def act_sync(self, obs: Observation) -> AgentAction:
        """Synchronous version of act method."""
        self.recv_message('Environment', obs)
        self._ensure_goal_sync()

        if self._only_none_action(obs.available_actions):
            return AgentAction(action_type='none', argument='')

        return self._model_provider.generate_action(
            history=self._history_text(self.inbox),
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=self.goal,
            script_like=self.script_like,
        )

    async def act(self, obs: Observation) -> AgentAction:
        """Asynchronous version of act method."""
        self.recv_message('Environment', obs)
        await self._ensure_goal()

        if self._only_none_action(obs.available_actions):
            return AgentAction(action_type='none', argument='')

        return await self._model_provider.agenerate_action(
            history=self._history_text(self.inbox),
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=self.goal,
            script_like=self.script_like,
        )

    def _ensure_goal_sync(self) -> None:
        """Synchronous version of _ensure_goal method."""
        if self._goal is not None:
            return
        background = self._first_message_text() or ''
        self._goal = self._model_provider.generate_goal(background=background)

    async def _ensure_goal(self) -> None:
        """Asynchronous version of _ensure_goal method."""
        if self._goal is not None:
            return
        background = self._first_message_text() or ''
        self._goal = await self._model_provider.agenerate_goal(background=background)

    @property
    def uses_provider(self) -> bool:
        return True

    @property
    def effective_model_name(self) -> str:
        return self._model_provider._get_agenerate_model_name()

    @property
    def provider_type(self) -> str:
        return f'{self._model_provider.__class__.__name__}({self._model_provider.type})'

    def set_model_provider(self, provider: 'BaseModelProvider') -> None:
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
