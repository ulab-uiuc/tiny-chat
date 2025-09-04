import json
import os
from pathlib import Path
from typing import Any

# Default prompt templates
DEFAULT_TEMPLATES = {
    "BAD_OUTPUT_REFORMAT": """
Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
Original string: {ill_formed_output}

Format instructions: {format_instructions}

Please only generate the JSON:
""",
    "ENV_PROFILE": """Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
Examples:
{examples}
Inspirational prompt: {inspiration_prompt}
Please use the following format:
{format_instructions}
""",
    "RELATIONSHIP_PROFILE": """Please generate relationship between two agents based on the agents' profiles below. Note that you generate
{agent_profile}
Please use the following format:
{format_instructions}
""",
    "ACTION_SCRIPT": """
Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
{history}.
The script has proceeded to Turn #{turn_number}. Current available action types are
{action_list}.
Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{format_instructions}
""",
    "ACTION_NORMAL": """
Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
Note that {agent}'s goal is only visible to you.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
{history}.
You are at Turn #{turn_number}. Your available action types are
{action_list}.
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{format_instructions}
""",
    "SCRIPT_SINGLE_STEP": """Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

Here are the conversation background and history:
{background}
{history}

Remember that you are an independent scriptwriter and should finish the script by yourself.
The output should only contain the script following the format instructions, with no additional comments or text.

Here are the format instructions:
{format_instructions}""",
    "SCRIPT_FULL": """
Please write the script between two characters based on their social goals with a maximum of 20 turns.

{background}
Your action should follow the given format:
{format_instructions}
Remember that you are an independent scriptwriter and should finish the script by yourself.
The output should only contain the script following the format instructions, with no additional comments or text.""",
    "INIT_PROFILE": """Please expand a fictional background for {name}. Here is the basic information:
    {name}'s age: {age}
    {name}'s gender identity: {gender_identity}
    {name}'s pronouns: {pronoun}
    {name}'s occupation: {occupation}
    {name}'s big 5 personality traits: {bigfive}
    {name}'s moral Foundation: think {mft} is more important than others
    {name}'s Schwartz portrait value: {schwartz}
    {name}'s decision-making style: {decision_style}
    {name}'s secret: {secret}
    Include the previous information in the background.
    Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
    For the personality and values (e.g., MBTI, moral foundation, and etc.),
    remember to use examples and behaviors in the person's life to demonstrate it.
    """,
    "FIRST_PERSON_NARRATIVE": """Please convert the following text into a first-person narrative.
e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
{text}""",
    "SECOND_PERSON_NARRATIVE": """Please convert the following text into a second-person narrative.
e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
{text}""",
    "GOAL": """Please generate your goal based on the background:
    {background}
    """,
}


class TemplateManager:
    """
    A class to manage all prompt templates, providing API for users to modify and customize prompts
    """

    def __init__(self, config_file: str | None = None):
        """
        Initialize TemplateManager

        Args:
            config_file: Configuration file path, if None uses default path
        """
        self.config_file = config_file or os.path.join(
            os.path.expanduser("~"), ".tiny_chat", "prompts.json"
        )
        self.templates = DEFAULT_TEMPLATES.copy()
        self._load_custom_templates()

    def _load_custom_templates(self) -> None:
        """Load custom templates from configuration file"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    custom_templates = json.load(f)
                    self.templates.update(custom_templates)
        except Exception as e:
            print(f"Warning: Unable to load custom prompt configuration: {e}")

    def _save_custom_templates(self) -> None:
        """Save custom templates to configuration file"""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Unable to save custom prompt configuration: {e}")

    def get_template(self, template_name: str) -> str:
        """
        Get the specified prompt template

        Args:
            template_name: Template name

        Returns:
            Prompt template string

        Raises:
            KeyError: If template doesn't exist
        """
        if template_name not in self.templates:
            raise KeyError(
                f"Template '{template_name}' does not exist. Available templates: {list(self.templates.keys())}"
            )
        return self.templates[template_name]

    def set_template(
        self, template_name: str, template_content: str, save: bool = True
    ) -> None:
        """
        Set or update prompt template

        Args:
            template_name: Template name
            template_content: Template content
            save: Whether to save to configuration file immediately
        """
        self.templates[template_name] = template_content
        if save:
            self._save_custom_templates()

    def add_template(
        self, template_name: str, template_content: str, save: bool = True
    ) -> None:
        """
        Add new prompt template

        Args:
            template_name: Template name
            template_content: Template content
            save: Whether to save to configuration file immediately
        """
        if template_name in self.templates:
            raise ValueError(
                f"Template '{template_name}' already exists. Use set_template() to update existing template."
            )

        self.set_template(template_name, template_content, save)

    def remove_template(self, template_name: str, save: bool = True) -> None:
        """
        Remove prompt template (can only remove custom templates, not default ones)

        Args:
            template_name: Template name
            save: Whether to save to configuration file immediately
        """
        if template_name in DEFAULT_TEMPLATES:
            raise ValueError(f"Cannot delete default template '{template_name}'")

        if template_name in self.templates:
            del self.templates[template_name]
            if save:
                self._save_custom_templates()
        else:
            raise KeyError(f"Template '{template_name}' does not exist")

    def list_templates(self) -> dict[str, str]:
        """
        List all available templates

        Returns:
            Dictionary containing all template names and content
        """
        return self.templates.copy()

    def list_template_names(self) -> list[str]:
        """
        List all template names

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def reset_to_defaults(self, save: bool = True) -> None:
        """
        Reset all templates to default values

        Args:
            save: Whether to save to configuration file immediately
        """
        self.templates = DEFAULT_TEMPLATES.copy()
        if save:
            self._save_custom_templates()

    def format_template(self, template_name: str, **kwargs: Any) -> str:
        """
        Format the specified template

        Args:
            template_name: Template name
            **kwargs: Formatting parameters

        Returns:
            Formatted prompt string
        """
        template = self.get_template(template_name)
        return template.format(**kwargs)

    def get_template_info(self, template_name: str) -> dict[str, Any]:
        """
        Get detailed information about a template

        Args:
            template_name: Template name

        Returns:
            Dictionary containing template information
        """
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' does not exist")

        template = self.templates[template_name]
        return {
            "name": template_name,
            "content": template,
            "is_default": template_name in DEFAULT_TEMPLATES,
            "length": len(template),
            "placeholder_count": template.count("{"),
        }


_prompt_manager = TemplateManager()


def get_template(template_name: str) -> str:
    """Get the specified prompt template (backward compatibility)"""
    return _prompt_manager.get_template(template_name)


def format_template(template_name: str, **kwargs: Any) -> str:
    """Format the specified template (backward compatibility)"""
    return _prompt_manager.format_template(template_name, **kwargs)


# Backward compatibility constants (but recommend using TemplateManager)
BAD_OUTPUT_REFORMAT_TEMPLATE = DEFAULT_TEMPLATES["BAD_OUTPUT_REFORMAT"]
ENV_PROFILE_TEMPLATE = DEFAULT_TEMPLATES["ENV_PROFILE"]
RELATIONSHIP_PROFILE_TEMPLATE = DEFAULT_TEMPLATES["RELATIONSHIP_PROFILE"]
ACTION_SCRIPT_TEMPLATE = DEFAULT_TEMPLATES["ACTION_SCRIPT"]
ACTION_NORMAL_TEMPLATE = DEFAULT_TEMPLATES["ACTION_NORMAL"]
SCRIPT_SINGLE_STEP_TEMPLATE = DEFAULT_TEMPLATES["SCRIPT_SINGLE_STEP"]
SCRIPT_FULL_TEMPLATE = DEFAULT_TEMPLATES["SCRIPT_FULL"]
INIT_PROFILE_TEMPLATE = DEFAULT_TEMPLATES["INIT_PROFILE"]
FIRST_PERSON_NARRATIVE_TEMPLATE = DEFAULT_TEMPLATES["FIRST_PERSON_NARRATIVE"]
SECOND_PERSON_NARRATIVE_TEMPLATE = DEFAULT_TEMPLATES["SECOND_PERSON_NARRATIVE"]
GOAL_TEMPLATE = DEFAULT_TEMPLATES["GOAL"]
