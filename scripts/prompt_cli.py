import argparse
import os
import sys

from tiny_chat.utils.template import TemplateManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Tiny Chat Prompt Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # List all templates
  python scripts/prompt_cli.py list

  # View specific template
  python scripts/prompt_cli.py show ACTION_SCRIPT

  # Modify template
  python scripts/prompt_cli.py set ACTION_SCRIPT "new prompt content"

  # Add new template
  python scripts/prompt_cli.py add MY_TEMPLATE "custom prompt content"

  # Remove custom template
  python scripts/prompt_cli.py remove MY_TEMPLATE

  # Reset to defaults
  python scripts/prompt_cli.py reset
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # list command
    list_parser = subparsers.add_parser('list', help='List all templates')
    list_parser.add_argument(
        '--names-only', action='store_true', help='Show only template names'
    )

    # show command
    show_parser = subparsers.add_parser('show', help='Show specific template')
    show_parser.add_argument('template_name', help='Template name')
    show_parser.add_argument(
        '--info', action='store_true', help='Show detailed information'
    )

    # set command
    set_parser = subparsers.add_parser('set', help='Set or update template')
    set_parser.add_argument('template_name', help='Template name')
    set_parser.add_argument('content', help='Template content')
    set_parser.add_argument(
        '--no-save', action='store_true', help='Do not save to config file'
    )

    # add command
    add_parser = subparsers.add_parser('add', help='Add new template')
    add_parser.add_argument('template_name', help='Template name')
    add_parser.add_argument('content', help='Template content')
    add_parser.add_argument(
        '--no-save', action='store_true', help='Do not save to config file'
    )

    # remove command
    remove_parser = subparsers.add_parser('remove', help='Remove custom template')
    remove_parser.add_argument('template_name', help='Template name')
    remove_parser.add_argument(
        '--no-save', action='store_true', help='Do not save to config file'
    )

    # reset command
    reset_parser = subparsers.add_parser('reset', help='Reset to default templates')
    reset_parser.add_argument(
        '--no-save', action='store_true', help='Do not save to config file'
    )

    # edit command
    edit_parser = subparsers.add_parser('edit', help='Edit template in editor')
    edit_parser.add_argument('template_name', help='Template name')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create prompt manager
    manager = TemplateManager()

    try:
        if args.command == 'list':
            handle_list(manager, args)
        elif args.command == 'show':
            handle_show(manager, args)
        elif args.command == 'set':
            handle_set(manager, args)
        elif args.command == 'add':
            handle_add(manager, args)
        elif args.command == 'remove':
            handle_remove(manager, args)
        elif args.command == 'reset':
            handle_reset(manager, args)
        elif args.command == 'edit':
            handle_edit(manager, args)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


def handle_list(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle list command"""
    if args.names_only:
        templates = manager.list_template_names()
        for name in templates:
            print(name)
    else:
        templates_dict = manager.list_templates()
        print('Available prompt templates:')
        print('=' * 50)
        for name, content in templates_dict.items():
            is_default = name in manager.templates and name in manager.templates
            status = '[Default]' if is_default else '[Custom]'
            print(f'{name} {status}')
            print(f'Length: {len(content)} characters')
            print(f"Placeholders: {content.count('{')} count")
            print('-' * 30)


def handle_show(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle show command"""
    template_name = args.template_name

    if args.info:
        info = manager.get_template_info(template_name)
        print(f"Template name: {info['name']}")
        print(f"Is default template: {info['is_default']}")
        print(f"Content length: {info['length']} characters")
        print(f"Placeholder count: {info['placeholder_count']}")
        print('\nTemplate content:')
        print('=' * 50)
        print(info['content'])
    else:
        content = manager.get_template(template_name)
        print(content)


def handle_set(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle set command"""
    template_name = args.template_name
    content = args.content
    save = not args.no_save

    manager.set_template(template_name, content, save)
    print(f"Updated template '{template_name}'")
    if save:
        print('Saved to configuration file')


def handle_add(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle add command"""
    template_name = args.template_name
    content = args.content
    save = not args.no_save

    manager.add_template(template_name, content, save)
    print(f"Added new template '{template_name}'")
    if save:
        print('Saved to configuration file')


def handle_remove(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle remove command"""
    template_name = args.template_name
    save = not args.no_save

    manager.remove_template(template_name, save)
    print(f"Removed template '{template_name}'")
    if save:
        print('Saved to configuration file')


def handle_reset(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle reset command"""
    save = not args.no_save

    manager.reset_to_defaults(save)
    print('Reset all templates to default values')
    if save:
        print('Saved to configuration file')


def handle_edit(manager: TemplateManager, args: argparse.Namespace) -> None:
    """Handle edit command"""
    template_name = args.template_name

    try:
        import os
        import subprocess
        import tempfile

        # Get current template content
        try:
            current_content = manager.get_template(template_name)
        except KeyError:
            current_content = ''

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(current_content)
            temp_file = f.name

        # Get editor
        editor = os.environ.get('EDITOR', 'vim')

        # Open editor
        subprocess.call([editor, temp_file])

        # Read modified content
        with open(temp_file) as f:
            new_content = f.read()

        # Delete temporary file
        os.unlink(temp_file)

        # Update template
        if new_content != current_content:
            manager.set_template(template_name, new_content)
            print(f"Updated template '{template_name}'")
        else:
            print('Content unchanged')

    except ImportError:
        print('Error: Unable to import subprocess module')
    except Exception as e:
        print(f'Error: Unable to open editor: {e}')


if __name__ == '__main__':
    main()
