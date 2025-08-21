import json
import os
from datetime import datetime


def save_conversation_to_json(
    agent_profile,
    environment_profile,
    conversation_history,
    evaluation,
    output_dir='conversation_logs',
):
    """
    Save the conversation details to a JSON file.

    :param agent_profile: Dictionary containing agent profile properties.
    :param environment_profile: Dictionary containing environment profile properties.
    :param conversation_history: List of conversation messages.
    :param evaluation: Dictionary containing evaluation results.
    :param output_dir: Directory to save the JSON file.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the data to save
    data = {
        'agent_profile': agent_profile if agent_profile is not None else {},
        'environment_profile': environment_profile
        if environment_profile is not None
        else '',
        'conversation_history': conversation_history
        if conversation_history is not None
        else '',
        'evaluation': evaluation if evaluation is not None else {},
    }

    # Generate a unique filename
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(output_dir, filename)

    # Write the data to a JSON file
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f'Conversation saved to {file_path}')
