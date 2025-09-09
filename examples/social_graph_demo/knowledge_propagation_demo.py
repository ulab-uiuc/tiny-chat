"""
Knowledge Propagation Demo with Private Conversations and True Knowledge Updates

This demo demonstrates true knowledge propagation where:
1. Agents have persistent memory that gets updated
2. Conversations are private (only participants know the content)
3. Knowledge spreads through the social network via direct interactions
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from demo_knowledge_evaluator import DemoKnowledgeEvaluator
from knowledge_agents import KnowledgeAgent
from private_conversation_environment import PrivateConversationEnvironment

from tiny_chat import BaseAgentProfile
from tiny_chat.server.config import ModelProviderConfig
from tiny_chat.server.providers import ModelProviderFactory


async def create_knowledge_agent(
    name: str,
    age: int,
    occupation: str,
    initial_knowledge: list[str] = None,
    goal: str = None,
) -> KnowledgeAgent:
    """Create a knowledge agent with proper model provider."""

    model_config = ModelProviderConfig(
        name='gpt-4o-mini', type='openai', temperature=0.7
    )
    model_provider = ModelProviderFactory.create_provider(model_config)

    profile = BaseAgentProfile(
        first_name=name.split()[0],
        last_name=name.split()[-1] if len(name.split()) > 1 else '',
        age=age,
        occupation=occupation,
        speaking_id=0,
        personality_and_values=f'Curious {occupation.lower()} interested in AI developments',
    )

    agent = KnowledgeAgent(
        agent_name=name,
        agent_profile=profile,
        model_provider=model_provider,
        initial_knowledge=initial_knowledge or [],
    )

    if goal:
        agent.goal = goal

    return agent


async def setup_sam_altman_scenario():
    """Set up the Sam Altman GPT-5 knowledge propagation scenario."""

    print('Setting up Sam Altman GPT-5 Knowledge Propagation Scenario')
    print('=' * 60)

    sam = await create_knowledge_agent(
        name='Sam Altman',
        age=39,
        occupation='CEO of OpenAI',
        initial_knowledge=['GPT-5 will be released in May 2025'],
        goal='Share information about GPT-5 release with researchers when appropriate',
    )

    alice = await create_knowledge_agent(
        name='Alice Chen',
        age=32,
        occupation='AI Researcher',
        goal='Learn about upcoming AI releases and share with colleagues',
    )

    bob = await create_knowledge_agent(
        name='Bob Wilson',
        age=28,
        occupation='Tech Journalist',
        goal='Gather information about AI releases for reporting',
    )

    reporter = await create_knowledge_agent(
        name='Sarah Reporter',
        age=35,
        occupation='Science Reporter',
        goal='Investigate AI release timelines for news stories',
    )

    social_network = {
        'Sam Altman': ['Alice Chen'],
        'Alice Chen': ['Sam Altman', 'Bob Wilson'],
        'Bob Wilson': ['Alice Chen', 'Sarah Reporter'],
        'Sarah Reporter': ['Bob Wilson'],
    }

    agents = {
        'Sam Altman': sam,
        'Alice Chen': alice,
        'Bob Wilson': bob,
        'Sarah Reporter': reporter,
    }

    return agents, social_network


async def run_evaluation(agents: dict, conversation_logs: list):
    """Run GPT-5 knowledge evaluation on the results."""

    evaluator = DemoKnowledgeEvaluator(
        target_knowledge='GPT-5 will be released in May 2025',
        knowledge_source='Sam Altman',
        target_agents=['Alice Chen', 'Bob Wilson', 'Sarah Reporter'],
    )

    print('\n' + '=' * 50)
    print('KNOWLEDGE PROPAGATION EVALUATION')
    print('=' * 50)

    for agent_name, agent in agents.items():
        status = agent.get_knowledge_status()

        print(f'\n{agent_name}:')
        print(f"  Total Knowledge Items: {status['total_knowledge']}")

        if status['knowledge_items']:
            for item in status['knowledge_items']:
                print(f'  - {item}')
        else:
            print('  - No knowledge items')

    has_gpt5_knowledge = []
    for agent_name, agent in agents.items():
        if agent_name == 'Sam Altman':
            continue

        knowledge_items = agent.get_all_knowledge()
        has_knowledge = any(
            'gpt-5' in item.lower() or 'may 2025' in item.lower()
            for item in knowledge_items
        )

        if has_knowledge:
            has_gpt5_knowledge.append(agent_name)

    total_target_agents = len(agents) - 1
    propagation_rate = (
        len(has_gpt5_knowledge) / total_target_agents if total_target_agents > 0 else 0
    )

    print('\nPropagation Summary:')
    print('  Knowledge source: Sam Altman')
    print(f'  Target agents: {total_target_agents}')
    print(f'  Agents with knowledge: {len(has_gpt5_knowledge)} ({has_gpt5_knowledge})')
    print(f'  Propagation rate: {propagation_rate:.1%}')

    return {
        'agents_with_knowledge': has_gpt5_knowledge,
        'propagation_rate': propagation_rate,
        'conversation_logs': conversation_logs,
    }


async def main():
    """Main demo function."""

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Error: OPENAI_API_KEY environment variable not set')
        return

    agents, social_network = await setup_sam_altman_scenario()

    environment = PrivateConversationEnvironment(
        agents=agents, social_network=social_network, max_rounds=5
    )

    results = await environment.run_knowledge_propagation_simulation()

    evaluation_results = await run_evaluation(agents, results['conversation_logs'])

    print(f"\nSimulation completed in {results['total_rounds']} rounds")
    print(f"Final propagation rate: {evaluation_results['propagation_rate']:.1%}")


if __name__ == '__main__':
    asyncio.run(main())
