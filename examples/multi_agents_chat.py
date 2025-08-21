import asyncio
import os
import sys
from pathlib import Path

from tiny_chat.messages import TinyChatBackground
from tiny_chat.server.core import TinyChatServer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main() -> None:
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

    background = TinyChatBackground(
        scenario='Three friends catching up over coffee',
        agent_configs=agent_configs,
    )

    print('Starting multi-agent conversation...')
    print('=' * 50)

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
