from tiny_chat.generator import agenerate_action
from tiny_chat.messages import AgentAction, Message, Observation
from tiny_chat.profile import BaseAgentProfile


class LLMAgent:
    """LLM-based agent that uses OpenAI API - simplified version"""

    def __init__(
        self,
        agent_name: str,
        model: str = 'gpt-4o-mini',
        agent_number: int = 1,
        api_key: str | None = None,
        goal: str | None = None,
        agent_profile: BaseAgentProfile | None = None,
        script_like: bool = False,
    ):
        self.agent_name = agent_name
        self.model = model
        self.agent_number = agent_number
        self.api_key = api_key
        self._goal = goal or 'Have a natural conversation'
        self.agent_profile = agent_profile
        self.message_history: list[Message] = []
        self.script_like = script_like

    @property
    def goal(self) -> str:
        """Get the agent's goal"""
        return self._goal

    @goal.setter
    def goal(self, goal: str) -> None:
        """Set the agent's goal"""
        self._goal = goal

    async def act(self, obs: Observation) -> AgentAction:
        self.receive_message('Environment', obs)

        if len(obs.available_actions) == 1 and 'none' in obs.available_actions:
            return AgentAction(action_type='none', argument='')
        else:
            action = await agenerate_action(
                model_name=self.model,
                history=self._build_context(obs),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
            )
            return action

    def _build_context(self, observation: Observation) -> str:
        """Build context string from message history and observation"""
        context_parts = []

        # Add agent profile information if available
        if self.agent_profile:
            context_parts.append(
                f'Agent Profile: {self.agent_profile.to_natural_language()}'
            )

        # Add background if available
        if self.goal:
            context_parts.append(f'Your goal: {self.goal}')

        # Add message history
        for msg in self.message_history:
            context_parts.append(f'{msg.to_natural_language()}')

        # Add current observation
        context_parts.append(observation.to_natural_language())

        return '\n'.join(context_parts)

    def reset(self) -> None:
        """Reset agent state"""
        self.message_history.clear()

    def receive_message(self, source: str, message: Message) -> None:
        """Receive a message and add to history"""
        self.message_history.append(message)


class HumanAgent:
    pass
