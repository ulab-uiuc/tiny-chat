#!/usr/bin/env python3
"""
Simple script to run a multi-agent chat conversation
Usage: python scripts/run_chat.py
"""

import asyncio
import os
import sys
from pathlib import Path

import yaml
from tiny_chat.messages import TwoAgentChatBackground
from tiny_chat.server import TinyChatServer

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main():
    """Run a simple multi-agent conversation"""

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. Some features may not work.')

    # Create chat server
    server = TinyChatServer(api_key=api_key)

    # Load configuration from YAML file
    config_path = Path(__file__).parent / 'two_agents_chat.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    agent_configs = [
        {
            'name': config['agent_configs'][0]['name'],
            'agent_number': config['agent_configs'][0]['agent_number'],
            'type': config['agent_configs'][0]['type'],
            'model': config['agent_configs'][0]['model'],
            'source': config['agent_configs'][0]['source'],
            'goal': config['agent_configs'][0]['goal'],
            'custom_profile': config['agent_configs'][0]['custom_profile'],
        },
        {
            'name': config['agent_configs'][1]['name'],
            'agent_number': config['agent_configs'][1]['agent_number'],
            'type': config['agent_configs'][1]['type'],
            'model': config['agent_configs'][1]['model'],
            'source': config['agent_configs'][1]['source'],
            'goal': config['agent_configs'][1]['goal'],
            'custom_profile': config['agent_configs'][1]['custom_profile'],
        },
    ]

    # Create a simple background
    background = TwoAgentChatBackground(
        scenario=config['background_settings']['scenario'],
        p1_background=config['background_settings']['p1_background'],
        p2_background=config['background_settings']['p2_background'],
        p1_goal=config['background_settings']['p1_goal'],
        p2_goal=config['background_settings']['p2_goal'],
        p1_name=config['background_settings']['p1_name'],
        p2_name=config['background_settings']['p2_name'],
    )
    general_settings = {
        'use_same_model': config['general_settings']['use_same_model'],
        'max_turns': config['general_settings']['max_turns'],
    }

    print('Starting multi-agent conversation...')
    print('=' * 50)

    # Run the conversation
    await server.two_agent_run_conversation(
        agent_configs=agent_configs,
        background=background,
        general_settings=general_settings,
        enable_evaluation=True,
    )


if __name__ == '__main__':
    asyncio.run(main())
