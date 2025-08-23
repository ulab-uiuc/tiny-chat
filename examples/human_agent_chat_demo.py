import asyncio
import os
import sys
from pathlib import Path

from tiny_chat import TinyChatBackground, create_server

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main() -> None:
    """Run a 3-agent conversation with one HumanAgent + two LLMAgents."""

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. LLM agents may not work as expected.')

    agent_configs = [
        {
            'name': 'Alice',
            'type': 'human',
            # You can optionally set a "goal" that will be displayed to the human
            'goal': 'Talk about weekend hiking plans',
        },
        {
            'name': 'Bob',
            'type': 'llm',
            'goal': 'Share thoughts on a new sci-fi book',
            # Optional: override model if your server supports this
            # "model_name": "gpt-4o-mini",
        },
        {
            'name': 'Carol',
            'type': 'llm',
            'goal': 'Discuss travel experiences and make everyone laugh',
            # "model_name": "gpt-4o-mini",
        },
    ]

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

    print('Starting human + bots conversation...')
    print('=' * 50)
    print(
        '\nHOW TO USE (HumanAgent):\n'
        '- Type multiple lines and press Enter on an empty line to send.\n'
        '- Or type commands like: speak: ... | action: ... | non-verbal communication: ...\n'
        '- Type /leave to leave the conversation.\n'
        '- Type /none to skip your turn.\n'
    )

    # Run conversation via server (kept same API as your example)
    async with create_server() as server:
        episode_log = await server.run_conversation(
            agent_configs=agent_configs,
            background=background,
            action_order='simultaneous',  # human + 2 llm act each turn
            max_turns=2,
            enable_evaluation=True,
            return_log=True,
        )

    # Print a brief summary
    if episode_log:
        print('\n=== Episode Log Created ===')
        print(f'Environment: {episode_log.environment}')
        print(f'Agents: {episode_log.agents}')
        print(f'Rewards: {episode_log.rewards}')
        try:
            print(f'Average Score: {episode_log.get_average_score():.2f}')
        except Exception:
            pass


if __name__ == '__main__':
    asyncio.run(main())
