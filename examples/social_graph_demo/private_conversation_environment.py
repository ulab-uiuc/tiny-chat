"""
Private conversation environment that supports knowledge propagation through direct agent interactions.
Only participants in a conversation can see the messages.
"""

import asyncio
from typing import Any

from knowledge_agents import KnowledgeAgent

from tiny_chat import TinyChatEnvironment


class PrivateConversationEnvironment:
    """
    Environment that manages private conversations between agents.
    Knowledge propagates only through direct interactions.
    """

    def __init__(
        self,
        agents: dict[str, KnowledgeAgent],
        social_network: dict[str, list[str]],
        max_rounds: int = 10,
    ):
        self.agents = agents
        self.social_network = social_network
        self.max_rounds = max_rounds
        self.current_round = 0
        self.conversation_logs = []

    async def run_knowledge_propagation_simulation(self) -> dict[str, Any]:
        """
        Run the knowledge propagation simulation through private conversations.
        """
        print('Starting knowledge propagation simulation')
        print(f'Social network: {self.social_network}')

        print('\n=== Initial Knowledge State ===')
        for agent_name, agent in self.agents.items():
            status = agent.get_knowledge_status()
            print(f"{agent_name}: {status['total_knowledge']} items")
            for item in status['knowledge_items']:
                print(f'  - {item}')

        for round_num in range(self.max_rounds):
            self.current_round = round_num + 1
            print(f'\n=== Round {self.current_round} ===')

            any_new_knowledge = await self._conduct_round()

            if not any_new_knowledge:
                print('No new knowledge shared this round. Ending simulation.')
                break

        print('\n=== Final Knowledge State ===')
        final_state = {}
        for agent_name, agent in self.agents.items():
            status = agent.get_knowledge_status()
            final_state[agent_name] = status
            print(f"{agent_name}: {status['total_knowledge']} items")
            for item in status['knowledge_items']:
                print(f'  - {item}')

        return {
            'final_knowledge_state': final_state,
            'total_rounds': self.current_round,
            'conversation_logs': self.conversation_logs,
        }

    async def _conduct_round(self) -> bool:
        """Conduct one round of private conversations."""
        any_new_knowledge = False
        round_conversations = []

        for agent_name, connected_agents in self.social_network.items():
            if agent_name not in self.agents:
                continue

            agent = self.agents[agent_name]

            for connected_agent_name in connected_agents:
                if connected_agent_name not in self.agents:
                    continue

                connected_agent = self.agents[connected_agent_name]

                knowledge_shared = await self._private_conversation(
                    agent, connected_agent, agent_name, connected_agent_name
                )

                if knowledge_shared:
                    any_new_knowledge = True
                    round_conversations.append(
                        {
                            'from': agent_name,
                            'to': connected_agent_name,
                            'knowledge_shared': knowledge_shared,
                        }
                    )

        self.conversation_logs.append(
            {'round': self.current_round, 'conversations': round_conversations}
        )

        return any_new_knowledge

    async def _private_conversation(
        self,
        agent1: KnowledgeAgent,
        agent2: KnowledgeAgent,
        agent1_name: str,
        agent2_name: str,
    ) -> bool:
        """Simulate a private conversation between two agents."""

        agent1_knowledge = set(agent1.get_all_knowledge())
        agent2_knowledge = set(agent2.get_all_knowledge())

        knowledge_to_share_1_to_2 = agent1_knowledge - agent2_knowledge
        knowledge_to_share_2_to_1 = agent2_knowledge - agent1_knowledge

        knowledge_shared = False

        if knowledge_to_share_1_to_2:
            shared_item = next(iter(knowledge_to_share_1_to_2))
            agent2.add_knowledge(shared_item)
            print(f'{agent1_name} shared with {agent2_name}: {shared_item}')
            knowledge_shared = True

        if knowledge_to_share_2_to_1:
            shared_item = next(iter(knowledge_to_share_2_to_1))
            agent1.add_knowledge(shared_item)
            print(f'{agent2_name} shared with {agent1_name}: {shared_item}')
            knowledge_shared = True

        await asyncio.sleep(0.1)

        return knowledge_shared


class PrivateConversationTinyChatEnvironment(TinyChatEnvironment):
    """
    TinyChatEnvironment that enforces private conversations based on neighbor_map.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保设置neighbor_map
        self.neighbor_map = getattr(self, 'neighbor_map', {})

    def _get_visible_agents(self, agent_name: str) -> list[str]:
        """Get list of agents visible to the given agent based on neighbor_map."""
        if hasattr(self, 'neighbor_map') and self.neighbor_map:
            return self.neighbor_map.get(agent_name, [])
        else:
            return [name for name in self.agents.keys() if name != agent_name]

    def _last_turn_text_for(self, agent_name: str) -> str:
        """Generate text representation of the last turn for a specific agent."""
        if not self.inbox:
            return ''

        visible_agents = self._get_visible_agents(agent_name)

        last_turn_messages = []
        for from_agent, message in self.inbox[-len(self.agents) :]:
            if (
                from_agent == agent_name
                or from_agent in visible_agents
                or from_agent == 'Environment'
            ):
                last_turn_messages.append(
                    f'{from_agent}: {message.to_natural_language()}'
                )

        return '\n'.join(last_turn_messages) if last_turn_messages else ''
