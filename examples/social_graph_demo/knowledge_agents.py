"""
Knowledge-aware agents that can store, share, and propagate information through social networks.
"""

from typing import Any

from tiny_chat import AgentAction, BaseAgentProfile, LLMAgent, Observation


class KnowledgeAgent(LLMAgent):
    """An agent capable of storing and sharing knowledge through memory."""

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: BaseAgentProfile | dict[str, Any] | None = None,
        profile_jsonl_path: str | None = None,
        model_provider: Any = None,
        script_like: bool = False,
        initial_knowledge: list[str] | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            profile_jsonl_path=profile_jsonl_path,
            model_provider=model_provider,
            script_like=script_like,
        )

        self.memory: list[str] = initial_knowledge or []

    def add_knowledge(self, knowledge: str) -> None:
        """Add new knowledge to agent's memory."""
        if knowledge and knowledge not in self.memory:
            self.memory.append(knowledge)

    def get_all_knowledge(self) -> list[str]:
        """Get all knowledge stored in memory."""
        return self.memory.copy()

    def search_knowledge(self, query: str) -> list[str]:
        """Search for relevant knowledge based on query."""
        relevant_knowledge = []
        query_lower = query.lower()

        for knowledge in self.memory:
            if any(keyword in knowledge.lower() for keyword in query_lower.split()):
                relevant_knowledge.append(knowledge)

        return relevant_knowledge

    async def act(self, obs: Observation) -> AgentAction:
        """Enhanced act method that considers knowledge sharing and learning."""
        self.recv_message('Environment', obs)
        await self._ensure_goal()

        if self._only_none_action(obs.available_actions):
            return AgentAction(action_type='none', argument='')

        await self._learn_from_observation(obs)

        enhanced_history = self._build_enhanced_history()

        action = await self._model_provider.agenerate_action(
            history=enhanced_history,
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=self.goal,
            script_like=self.script_like,
        )

        await self._learn_from_own_action(action)

        return action

    def _build_enhanced_history(self) -> str:
        """Build conversation history enhanced with agent's knowledge."""
        base_history = self._history_text(self.inbox)

        if self.memory:
            knowledge_context = 'My current knowledge: ' + '; '.join(self.memory)
            return f'{knowledge_context}\n\n{base_history}'

        return base_history

    def _extract_knowledge_from_conversation(self, conversation: str) -> list[str]:
        """Extract potential knowledge from conversation text."""
        knowledge_indicators = [
            'will be released',
            'is scheduled for',
            'announced that',
            'confirmed that',
            'stated that',
        ]

        extracted = []
        sentences = conversation.split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in knowledge_indicators):
                if len(sentence) > 20:
                    extracted.append(sentence)

        return extracted

    async def _learn_from_observation(self, obs: Observation) -> None:
        """Learn knowledge from observation (what other agents said)."""
        obs_text = obs.to_natural_language()
        new_knowledge = self._extract_knowledge_from_conversation(obs_text)

        for knowledge in new_knowledge:
            print(f'[{self.agent_name}] Learned: {knowledge}')
            self.add_knowledge(knowledge)

    async def _learn_from_own_action(self, action: AgentAction) -> None:
        """Learn knowledge from own action (what I just said)."""
        if action.action_type == 'speak':
            action_text = action.to_natural_language()
            new_knowledge = self._extract_knowledge_from_conversation(action_text)

            for knowledge in new_knowledge:
                if knowledge not in self.memory:
                    print(f'[{self.agent_name}] Reinforced knowledge: {knowledge}')
                    self.add_knowledge(knowledge)
