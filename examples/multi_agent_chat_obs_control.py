import asyncio
from pathlib import Path

from tiny_chat import ConfigManager, TinyChatBackground, TinyChatServer


async def tiny_chat_demo_conversation():
    """Demonstrate conversation using Agents with local observation control"""

    print('\n\nTiny-Chat Conversation Demo (with local observation control)')
    print('=' * 60)

    # Load configuration from config/environments/demo.yaml
    config_path = Path('config/environments/demo.yaml')
    config_manager = ConfigManager(config_path)
    server_config = config_manager.load_config()

    print(f'Using configuration from: {config_path}')
    print(f'Available models: {list(server_config.models.keys())}')

    agent_configs = [
        {
            'name': 'Alice',
            'model_provider': 'model1',
            'goal': 'Start an interesting conversation about AI',
            'speaking_id': 0,
        },
        {
            'name': 'Bob',
            'model_provider': 'model2',
            'goal': 'Respond thoughtfully and ask questions',
            'speaking_id': 1,
        },
        {
            'name': 'Charlie',
            'model_provider': 'model1',
            'goal': 'Provide unique perspectives on the topic',
            'speaking_id': 2,
        },
    ]

    background = TinyChatBackground(
        scenario='Three AI assistants discussing the future of artificial intelligence',
        agent_configs=[
            {
                'name': 'Alice',
                'background': 'Alice is curious and likes to kick off topics.',
                'goal': 'Start an interesting conversation about AI',
            },
            {
                'name': 'Bob',
                'background': 'Bob enjoys thoughtful, probing questions.',
                'goal': 'Respond thoughtfully and ask questions',
            },
            {
                'name': 'Charlie',
                'background': 'Charlie brings creative, novel perspectives.',
                'goal': 'Provide unique perspectives on the topic',
            },
        ],
    )

    print('Starting conversation...')
    print(f'Scenario: {background.scenario}')

    obs_control = {
        'mode': 'local',
        'neighbor_map': {
            0: [1],
            1: [0, 2],
            2: [1],
        },
        'local_k': 1,
    }

    try:
        server = TinyChatServer(server_config)
        await server.initialize()

        episode_log = await server.run_conversation(
            agent_configs=agent_configs,
            background=background,
            max_turns=4,
            enable_evaluation=True,
            return_log=True,
            action_order='simultaneous',
            obs_control=obs_control,
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
    await tiny_chat_demo_conversation()


if __name__ == '__main__':
    asyncio.run(main())
