#!/usr/bin/env python3
"""
Simple script to run a multi-agent chat conversation
Usage: python scripts/run_chat.py
"""

import asyncio
import os

from tiny_chat.messages import ChatBackground
from tiny_chat.server import ChatServer


async def main():
    """Run a simple multi-agent conversation"""

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. Some features may not work.')

    # Create chat server
    server = ChatServer(api_key=api_key)

    # Define agent configurations
    agent_configs = [
        {
            'name': 'Alice',
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Be friendly and helpful in the conversation',
        },
        {
            'name': 'Bob',
            'type': 'llm',
            'model': 'gpt-4o-mini',
            'goal': 'Ask thoughtful questions and share interesting ideas',
        },
    ]

    # Create a simple background
    background = ChatBackground(
        scenario='Two friends meeting at a coffee shop',
        p1_background='Alice is a software engineer who loves hiking',
        p2_background='Bob is a teacher who enjoys reading science fiction',
        p1_goal='Have a pleasant conversation about weekend plans',
        p2_goal='Discuss recent books and outdoor activities',
        p1_name='Alice',
        p2_name='Bob',
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
