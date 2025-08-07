import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tiny_chat.messages import MultiAgentChatBackground
from tiny_chat.server import TinyChatServer


async def main():
    """Run a 3-agent conversation"""

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. Some features may not work.')

    # Create chat server
    server = TinyChatServer(api_key=api_key)

    # Define 3-agent configurations
    agent_configs = [
        {
            'name': 'Alice',
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Talk about weekend hiking plans',
        },
        {
            'name': 'Bob',
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Share thoughts on a new sci-fi book',
        },
        {
            'name': 'Carol',
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Discuss travel experiences and make everyone laugh',
        },
    ]

    # Create background object
    background = MultiAgentChatBackground(
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

    await server.multi_agent_run_conversation(
        agent_configs=agent_configs,
        background=background,
        action_order='simultaneous',  # sequential, round-robin, simultaneous, random
        max_turns=12,
        enable_evaluation=True,
    )


if __name__ == '__main__':
    asyncio.run(main())
