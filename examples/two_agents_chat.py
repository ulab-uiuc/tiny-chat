#!/usr/bin/env python3
"""
Simple script to run a multi-agent chat conversation
Usage: python scripts/run_chat.py
"""

import asyncio
import os
import sys
from pathlib import Path

from tiny_chat.messages import UnifiedChatBackground
from tiny_chat.utils.server import TinyChatServer

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main() -> None:
    """Run a simple multi-agent conversation"""

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. Some features may not work.')

    # Create chat server
    server = TinyChatServer(api_key=api_key)

    # Define agent configurations
    agent_configs = [
        {
            'name': 'Alice',
            'agent_number': 1,
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Be friendly and helpful in the conversation',
        },
        {
            'name': 'Bob',
            'agent_number': 2,
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Ask thoughtful questions and share interesting ideas',
        },
    ]

    background = UnifiedChatBackground(
        scenario='Two friends meeting at a coffee shop',
        agent_configs=[
            {
                'name': 'Alice',
                'background': 'Alice is a software engineer who loves hiking',
                'goal': 'Have a pleasant conversation about weekend plans',
            },
            {
                'name': 'Bob',
                'background': 'Bob is a teacher who enjoys reading science fiction',
                'goal': 'Discuss recent books and outdoor activities',
            },
        ],
    )

    print('Starting multi-agent conversation...')
    print('=' * 50)

    # Run the conversation
    await server.run_conversation(
        agent_configs=agent_configs,
        background=background,
        max_turns=10,
        enable_evaluation=True,
    )


if __name__ == '__main__':
    asyncio.run(main())
