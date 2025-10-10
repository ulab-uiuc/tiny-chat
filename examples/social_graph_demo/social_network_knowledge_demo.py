"""
Social Network Knowledge Propagation Benchmark

This demo implements a knowledge propagation scenario where agents share information
through a social network. The scenario features Sam Altman sharing GPT-5 release
information with other agents through social connections.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import glob
import json

from demo_knowledge_evaluator import DemoKnowledgeEvaluator

from tiny_chat import SimpleMessage, TinyChatBackground
from tiny_chat.server.core import create_server


async def run_custom_evaluation_from_logs(evaluator: DemoKnowledgeEvaluator):
    """Run custom evaluation on the most recent conversation log."""

    log_pattern = 'conversation_logs/conversation_*.json'
    log_files = glob.glob(log_pattern)

    if not log_files:
        print('No conversation logs found for evaluation')
        return

    latest_log = max(log_files, key=os.path.getctime)
    print(f'Evaluating conversation log: {latest_log}')

    try:
        with open(latest_log, encoding='utf-8') as f:
            conversation_data = json.load(f)

        evaluator_messages = []

        if 'conversation_history' in conversation_data:
            for msg_data in conversation_data['conversation_history']:
                try:
                    if (
                        isinstance(msg_data, dict)
                        and 'agent' in msg_data
                        and 'content' in msg_data
                    ):
                        source = msg_data['agent']
                        message_content = msg_data['content']

                        if message_content.startswith(
                            'said: "'
                        ) and message_content.endswith('"'):
                            message_content = message_content[7:-1]
                        elif message_content == 'did nothing':
                            continue

                        message = SimpleMessage(message=message_content)
                        evaluator_messages.append((source, message))

                except Exception as e:
                    print(f'Error processing message: {e}')
                    continue
        else:
            print('No conversation_history found in log file')
            return

        if not evaluator_messages:
            print('No evaluable messages found in conversation log')
            return

        print(f'Found {len(evaluator_messages)} evaluable messages:')
        for source, msg in evaluator_messages:
            content = msg.to_natural_language()
            preview = content[:60] + ('...' if len(content) > 60 else '')
            print(f'  {source}: {preview}')

        print(f'\nCalling evaluator with {len(evaluator_messages)} messages...')
        evaluation_results = await evaluator.__acall__(
            turn_number=len(evaluator_messages), messages=evaluator_messages
        )
        print(
            f'Evaluator completed. Results: {len(evaluation_results) if evaluation_results else 0} items'
        )

        print('\n' + '=' * 50)
        print('CUSTOM KNOWLEDGE EVALUATION')
        print('=' * 50)

        for metric_name, metric_data in evaluation_results:
            if isinstance(metric_data, tuple) and len(metric_data) == 2:
                score_info, comments = metric_data
                if isinstance(score_info, tuple) and len(score_info) == 2:
                    actual_metric_name, actual_score = score_info
                    print(f'\n{actual_metric_name.upper()}: {actual_score:.2f}/10')
                else:
                    print(f'\n{metric_name.upper()}: {score_info:.2f}/10')
                print('Comments:')
                print(comments)
            else:
                print(f'\n{metric_name.upper()}: {metric_data}')

    except Exception as e:
        print(f'Error during evaluation: {e}')
        import traceback

        traceback.print_exc()


async def main():
    """Main demo function."""

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Warning: OPENAI_API_KEY not set. Demo may not work properly.')
        return

    agent_configs = [
        {
            'name': 'Sam Altman',
            'type': 'llm',
            'goal': 'I am the CEO of OpenAI with insider knowledge. I know GPT-5 will be released in May 2025. I should share this information when Alice asks about upcoming AI developments.',
            'model_provider': 'gpt-4o-mini',
            'speaking_id': 0,
        },
        {
            'name': 'Alice Chen',
            'type': 'llm',
            'goal': 'I am an AI researcher. I MUST ask Sam Altman about upcoming AI releases first, then immediately share any information I learn with Bob. I will not leave until I complete both tasks.',
            'model_provider': 'gpt-4o-mini',
            'speaking_id': 1,
        },
        {
            'name': 'Bob Wilson',
            'type': 'llm',
            'goal': 'I am a tech journalist. I MUST wait for Alice to share GPT-5 information with me, then share this information with the Reporter. I will not leave until I complete both tasks.',
            'model_provider': 'gpt-4o-mini',
            'speaking_id': 2,
        },
        {
            'name': 'Reporter',
            'type': 'llm',
            'goal': 'I need GPT-5 release information for my story. I MUST wait for Bob to share information with me and ask follow-up questions to get complete details. I will not leave until I get the information.',
            'model_provider': 'gpt-4o-mini',
            'speaking_id': 3,
        },
    ]

    background = TinyChatBackground(
        scenario='A professional AI industry discussion where knowledge about GPT-5 release spreads through the network',
        agent_configs=[
            {
                'name': 'Sam Altman',
                'background': 'Sam Altman is the CEO of OpenAI with exclusive insider knowledge that GPT-5 will be released in May 2025. He is willing to share this information with AI researchers.',
                'goal': 'Share GPT-5 release information when asked by Alice about upcoming AI developments',
            },
            {
                'name': 'Alice Chen',
                'background': 'Alice is an AI researcher who MUST complete the knowledge chain. She will ask Sam about GPT-5, then immediately inform Bob. She stays until both tasks are done.',
                'goal': 'Ask Sam about GPT-5 releases, then actively share the information with Bob to continue the knowledge chain',
            },
            {
                'name': 'Bob Wilson',
                'background': 'Bob is a tech journalist who MUST wait for Alice to share GPT-5 information, then pass it to Reporter. He stays until both tasks are completed.',
                'goal': 'Receive GPT-5 information from Alice, then actively share it with Reporter to complete the knowledge propagation',
            },
            {
                'name': 'Reporter',
                'background': 'Reporter is investigating GPT-5 for breaking news. Must stay and actively seek information from Bob until getting complete GPT-5 release details.',
                'goal': 'Actively ask Bob for GPT-5 information and stay until receiving complete release timeline details',
            },
        ],
    )

    obs_control = {
        'mode': 'local',
        'neighbor_map': {
            'Sam Altman': ['Alice Chen'],
            'Alice Chen': ['Sam Altman', 'Bob Wilson'],
            'Bob Wilson': ['Alice Chen', 'Reporter'],
            'Reporter': ['Bob Wilson'],
        },
    }

    print('Social Network Knowledge Propagation Demo')
    print('Scenario: GPT-5 Release Information Spreading')
    print('-' * 50)
    print('Starting knowledge propagation scenario...')
    print('Social network topology:')
    print('  Sam Altman -> Alice Chen')
    print('  Alice Chen -> Sam Altman, Bob Wilson')
    print('  Bob Wilson -> Alice Chen, Reporter')
    print('  Reporter -> Bob Wilson')
    print()

    print('Expected knowledge propagation chain:')
    print(
        "1. Sam Altman (Source): Has exclusive knowledge 'GPT-5 will be released in May 2025'"
    )
    print('2. Alice Chen asks Sam about upcoming AI releases')
    print('3. Sam shares GPT-5 release information with Alice')
    print('4. Bob Wilson asks Alice about GPT-5 release dates')
    print('5. Alice shares the May 2025 information with Bob')
    print('6. Reporter asks Bob about GPT-5 timeline')
    print('7. Bob shares the release information with Reporter')
    print('8. Knowledge successfully propagates: Sam → Alice → Bob → Reporter')
    print()

    try:
        demo_evaluator = DemoKnowledgeEvaluator(
            target_knowledge='GPT-5 will be released in May 2025',
            knowledge_source='Sam Altman',
            target_agents=['Alice Chen', 'Bob Wilson', 'Reporter'],
        )

        print('Running conversation with private observation control...')

        async with create_server() as server:
            episode_log = await server.run_conversation(
                agent_configs=agent_configs,
                background=background,
                max_turns=12,
                enable_evaluation=True,
                return_log=True,
                action_order='sequential',
                obs_control=obs_control,
            )

            print('\nRunning custom knowledge evaluation...')
            await run_custom_evaluation_from_logs(demo_evaluator)

        print('\n' + '=' * 50)
        print('KNOWLEDGE PROPAGATION RESULTS')
        print('=' * 50)
        episode_length = getattr(
            episode_log, 'episode_length', len(episode_log.rewards)
        )
        print(f'Total conversation turns: {episode_length}')

        print('\nKnowledge propagation analysis:')
        print('Knowledge indicators found: See evaluation results above')
        print('Propagation success: See evaluation results above')

        print(
            f'\nEvaluation Scores: {[(score, data) for score, data in episode_log.rewards]}'
        )

        print('\nConversation log saved and can be reviewed for detailed analysis.')

    except Exception as e:
        print(f'Error running demo: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
