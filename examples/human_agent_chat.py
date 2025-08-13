#!/usr/bin/env python3
"""
Interactive chat script - Chat with AI agents
Usage: python scripts/interactive_chat.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from tiny_chat.agents import LLMAgent
from tiny_chat.messages import Observation, UnifiedChatBackground

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def interactive_chat() -> None:
    """Interactive chat with an AI agent"""

    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Error: OPENAI_API_KEY environment variable is required')
        return

    agent = LLMAgent(
        agent_name='AI Assistant',
        model_name='gpt-4o-mini',
    )

    _background = UnifiedChatBackground(
        scenario='A helpful AI assistant chatting with a human',
        agent_configs=[
            {
                'name': 'AI Assistant',
                'background': 'You are a helpful AI assistant',
                'goal': 'Provide helpful and engaging responses',
            },
            {
                'name': 'Human',
                'background': 'A human user who wants to chat',
                'goal': 'Have an interesting conversation',
            },
        ],
    )

    print("🤖 AI Assistant: Hello! I'm your AI assistant. How can I help you today?")
    print("(Type 'quit' or 'q' to exit)")
    print('-' * 50)

    turn_number = 1

    while True:
        user_input = input('👤 You: ').strip()

        if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
            print('🤖 AI Assistant: Goodbye! It was nice chatting with you!')
            break

        if not user_input:
            continue

        observation = Observation(
            last_turn=f'Human says: {user_input}',
            available_actions=['speak', 'none'],
            turn_number=turn_number,
        )

        try:
            action = await agent.act(observation)

            if action.action_type == 'speak':
                print(f'AI Assistant: {action.argument}')
            else:
                print('AI Assistant: [No response]')

        except Exception as e:
            print(f'AI Assistant: Sorry, I encountered an error: {e}')

        turn_number += 1
        print('-' * 50)


if __name__ == '__main__':
    asyncio.run(interactive_chat())
