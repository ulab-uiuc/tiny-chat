import asyncio

from tiny_chat.agents import LLMAgent
from tiny_chat.messages import TinyChatBackground
from tiny_chat.server.config import ModelProviderConfig, ServerConfig
from tiny_chat.server.core import TinyChatServer
from tiny_chat.server.providers import ModelProviderFactory


async def demo_conversation_with_different_models():
    """Demonstrate conversation using Agents with different models"""

    print('\n\nMulti-Model Conversation Demo')
    print('=' * 60)
    models = {
        'model0': ModelProviderConfig(
            name='gpt-4o-mini',
            type='openai',
            temperature=0.7,
        ),
        'model1': ModelProviderConfig(
            name='gpt-4o',
            type='openai',
            temperature=0.1,
        ),
    }

    agent_configs = [
        {
            'name': 'Alice',
            'model_name': 'model0',
            'goal': 'Start an interesting conversation about AI',
        },
        {
            'name': 'Bob',
            'model_name': 'model1',
            'goal': 'Respond thoughtfully and ask questions',
        },
    ]

    server_config = ServerConfig(
        models=models,
        default_model='model0',
        action_order='round-robin',
        available_action_types=['none', 'speak', 'leave'],
    )

    background = TinyChatBackground(
        scenario='Two AI assistants discussing the future of artificial intelligence',
        agent_configs=agent_configs,
    )

    print('Starting conversation...')
    print(f'Scenario: {background.scenario}')

    try:
        server = TinyChatServer(server_config)
        await server.initialize()

        episode_log = await server.run_conversation(
            agent_configs=agent_configs,
            background=background,
            max_turns=4,
            enable_evaluation=True,
            return_log=True,
        )

        if episode_log:
            total_turns = getattr(episode_log, 'episode_length', None)
            if total_turns is None:
                total_turns = (
                    len(episode_log.rewards)
                    if hasattr(episode_log, 'rewards')
                    else 'Unknown'
                )
            print(f'Total turns: {total_turns}')

    except Exception as e:
        print(f'Conversation failed: {e}')
        print('This is expected if API keys are not set')


async def main():
    """Run all demonstrations"""
    await demo_conversation_with_different_models()


if __name__ == '__main__':
    asyncio.run(main())
