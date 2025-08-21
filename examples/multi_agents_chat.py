import asyncio
import os
import sys
from pathlib import Path

from tiny_chat.messages import TinyChatBackground
from tiny_chat.server.core import TinyChatServer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main() -> None:
    """Run a 3-agent conversation"""

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. Some features may not work.')

    # Define 3-agent configurations
    agent_configs = [
        {
            'name': 'Alice',
            'type': 'llm',
            'goal': 'Talk about weekend hiking plans',
        },
        {
            'name': 'Bob',
            'type': 'llm',
            'goal': 'Share thoughts on a new sci-fi book',
        },
        {
            'name': 'Carol',
            'type': 'llm',
            'goal': 'Discuss travel experiences and make everyone laugh',
        },
    ]

    # Create background object
    background = TinyChatBackground(
        scenario='Three friends catching up over coffee',
        agent_configs=[
            {
                'name': 'Alice',
                'background': 'Alice is a software engineer who loves hiking',
                'goal': 'Talk about weekend hiking plans',
            },
            {
                'name': 'Bob',
                'background': 'Bob is a teacher who enjoys reading science fiction',
                'goal': 'Share thoughts on a new sci-fi book',
            },
            {
                'name': 'Carol',
                'background': 'Carol is a graphic designer who recently returned from a trip to Japan',
                'goal': 'Discuss travel experiences and make everyone laugh',
            },
        ],
    )

    print('Starting multi-agent conversation...')
    print('=' * 50)

    # Run the conversation using the new server architecture
    from tiny_chat.server.core import create_server

    async with create_server() as server:
        episode_log = await server.run_conversation(
            agent_configs=agent_configs,
            background=background,
            action_order='simultaneous',
            max_turns=2,
            enable_evaluation=True,
            return_log=True,
        )

    if episode_log:
        print('\n=== Episode Log Created ===')
        print(f'Environment: {episode_log.environment}')
        print(f'Agents: {episode_log.agents}')
        print(f'Rewards: {episode_log.rewards}')
        print(f'Average Score: {episode_log.get_average_score():.2f}')


if __name__ == '__main__':
    asyncio.run(main())
