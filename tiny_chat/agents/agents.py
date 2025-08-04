from typing import List, Optional
from tiny_chat.messages import Observation, AgentAction, Message
from tiny_chat.profile import AgentProfile
from tiny_chat.generator import generate_agent
from tiny_chat.generator import generate_env_profile


class LLMAgent:
    """LLM-based agent that uses OpenAI API - simplified version"""

    def __init__(
        self,
        agent_name: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        goal: Optional[str] = None,
        agent_profile: Optional[AgentProfile] = None,
    ):
        self.agent_name = agent_name
        self.model = model
        self.api_key = api_key
        self._goal = goal or "Have a natural conversation"
        self.agent_profile = agent_profile
        self.message_history: List[Message] = []

    @property
    def goal(self) -> str:
        """Get the agent's goal"""
        return self._goal

    @goal.setter
    def goal(self, goal: str) -> None:
        """Set the agent's goal"""
        self._goal = goal

    async def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            action = await generate_agent()
            return action

    def _build_context(self, observation: Observation) -> str:
        """Build context string from message history and observation"""
        context_parts = []

        # Add agent profile information if available
        if self.agent_profile:
            context_parts.append(
                f"Agent Profile: {self.agent_profile.to_natural_language()}"
            )

        # Add background if available
        if self.goal:
            context_parts.append(f"Your goal: {self.goal}")

        # Add message history
        for msg in self.message_history:
            context_parts.append(f"{msg.to_natural_language()}")

        # Add current observation
        context_parts.append(observation.to_natural_language())

        return "\n".join(context_parts)

    def reset(self) -> None:
        """Reset agent state"""
        self.message_history.clear()

    def receive_message(self, message: Message) -> None:
        """Receive a message and add to history"""
        self.message_history.append(message)
