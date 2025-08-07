import os
import sys

from tiny_chat.utils.template import TemplateManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def basic_usage_example():
    """Basic usage example"""
    print('=== Basic Usage Example ===')

    manager = TemplateManager()

    action_template = manager.get_template('ACTION_SCRIPT')
    print(f'ACTION_SCRIPT template length: {len(action_template)} characters')

    formatted = manager.format_template(
        'ACTION_SCRIPT',
        agent='Alice',
        history='Previous conversation history',
        turn_number=5,
        action_list='speak, listen, leave',
        format_instructions='JSON format',
    )
    print(f'Formatted template:\n{formatted[:200]}...')

    print()


def custom_template_example():
    """Custom template example"""
    print('=== Custom Template Example ===')

    manager = TemplateManager()

    # Add custom template
    custom_prompt = """
You are a professional {role}. Please provide advice based on the following information:

Background: {background}
Issue: {issue}

Please respond in a {style} tone and give specific recommendations.
"""

    manager.add_template('CUSTOM_ADVICE', custom_prompt)

    # Use custom template
    advice = manager.format_template(
        'CUSTOM_ADVICE',
        role='counselor',
        background='25-year-old recent graduate',
        issue='work stress and anxiety',
        style='gentle and professional',
    )

    print('Custom counseling template:')
    print(advice)
    print()


def template_management_example():
    """Template management example"""
    print('=== Template Management Example ===')

    manager = TemplateManager()

    print('All available templates:')
    for name in manager.list_template_names():
        info = manager.get_template_info(name)
        status = 'Default' if info['is_default'] else 'Custom'
        print(f'  {name} ({status})')

    print()

    original_content = manager.get_template('GOAL')
    modified_content = original_content.replace(
        'Please generate your goal based on the background:',
        'Please generate your objective based on the following background:',
    )

    manager.set_template('GOAL', modified_content, save=False)

    print('Modified GOAL template:')
    print(manager.get_template('GOAL'))
    print()


def batch_operations_example():
    """Batch operations example"""
    print('=== Batch Operations Example ===')

    manager = TemplateManager()

    templates_to_add = {
        'GREETING': "Hello, I'm {name}, nice to meet you!",
        'FAREWELL': 'Goodbye, {name}! Hope to see you again.',
        'QUESTION': 'I have a question for you: {question}',
        'COMPLIMENT': "You're really great, {compliment}!",
    }

    for name, content in templates_to_add.items():
        try:
            manager.add_template(name, content, save=False)
            print(f'Added template: {name}')
        except ValueError:
            print(f'Template {name} already exists, skipping')

    print()

    test_data = {
        'name': 'Sarah',
        'question': 'How are you doing today?',
        'compliment': 'your smile is very warm',
    }

    for template_name in ['GREETING', 'QUESTION', 'COMPLIMENT']:
        try:
            result = manager.format_template(template_name, **test_data)
            print(f'{template_name}: {result}')
        except KeyError as e:
            print(f'Error formatting {template_name}: {e}')

    print()


def error_handling_example():
    """Error handling example"""
    print('=== Error Handling Example ===')

    manager = TemplateManager()

    try:
        manager.get_template('NONEXISTENT_TEMPLATE')
    except KeyError as e:
        print(f'Expected error: {e}')

    try:
        manager.remove_template('ACTION_SCRIPT')
    except ValueError as e:
        print(f'Expected error: {e}')

    try:
        manager.add_template('ACTION_SCRIPT', 'new content')
    except ValueError as e:
        print(f'Expected error: {e}')

    print()


def configuration_example():
    """Configuration management example"""
    print('=== Configuration Management Example ===')

    custom_config = '/tmp/custom_prompts.json'
    manager = TemplateManager(config_file=custom_config)

    manager.add_template('CUSTOM_TEST', 'This is a test template: {test_var}')

    manager2 = TemplateManager(config_file=custom_config)

    try:
        content = manager2.get_template('CUSTOM_TEST')
        print(f'Template loaded from config file: {content}')
    except KeyError:
        print('Template not properly saved to config file')

    import os

    if os.path.exists(custom_config):
        os.remove(custom_config)

    print()


def rest_api_example():
    """REST API usage example"""
    print('=== REST API Usage Example ===')
    print('To use the REST API:')
    print('1. Start the API server:')
    print('   python -m tiny_chat.utils.prompt_api_server')
    print()
    print('2. Example API calls:')
    print('   # List all templates')
    print('   curl http://localhost:5000/api/prompts')
    print()
    print('   # Get specific template')
    print('   curl http://localhost:5000/api/prompts/ACTION_SCRIPT')
    print()
    print('   # Add new template')
    print('   curl -X POST http://localhost:5000/api/prompts \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"name": "NEW_TEMPLATE", "content": "Hello {name}!"}\'')
    print()
    print('   # Update template')
    print('   curl -X PUT http://localhost:5000/api/prompts/ACTION_SCRIPT \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"content": "Updated content"}\'')
    print()
    print('   # Format template')
    print('   curl -X POST http://localhost:5000/api/prompts/ACTION_SCRIPT/format \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"parameters": {"agent": "Alice", "turn_number": 1}}\'')
    print()


def main():
    """Main function"""
    print('Tiny Chat Prompt API Usage Examples')
    print('=' * 50)

    basic_usage_example()
    custom_template_example()
    template_management_example()
    batch_operations_example()
    error_handling_example()
    configuration_example()
    rest_api_example()

    print('Examples completed!')
    print('\nTips:')
    print('1. Use TemplateManager() to manage prompt templates')
    print('2. Use add_template() to add new templates')
    print('3. Use set_template() to modify existing templates')
    print('4. Use format_template() to format templates')
    print('5. Use CLI tool for command-line management')
    print('6. Use REST API for web-based management')


if __name__ == '__main__':
    main()
