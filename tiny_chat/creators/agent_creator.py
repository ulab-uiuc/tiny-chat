from typing import Any

from pydantic import validate_call

from tiny_chat.generator.generate import agenerate
from tiny_chat.generator.output_parsers import PydanticOutputParser
from tiny_chat.profiles.agent_profile import BaseAgentProfile
from tiny_chat.utils.template import TemplateManager

from .base_creator import BaseCreator


class AgentCreator(BaseCreator):
    """Creator for agent profiles with LLM enhancement"""

    def __init__(self, model_name: str = 'gpt-4o-mini', output_dir: str = 'data'):
        super().__init__(model_name, output_dir)
        self.template_manager = TemplateManager()

    def get_default_filename(self) -> str:
        """Get default output filename"""
        return 'agent_profiles.jsonl'

    @validate_call
    async def create_profile(
        self,
        partial_info: dict[str, Any],
        output_file: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> BaseAgentProfile:
        """
        Create an agent profile with LLM enhancement

        Args:
            partial_info: Partial information provided by user
            output_file: Output file name (defaults to agent_profiles.jsonl)
            temperature: LLM temperature for generation
            **kwargs: Additional parameters

        Returns:
            Created BaseAgentProfile object
        """
        unique_id = self.generate_unique_id('agent')

        provided_info_parts = []
        if partial_info:
            provided_info_parts.append('Provided information:')
            for key, value in partial_info.items():
                if value is not None and value != '' and value != []:
                    provided_info_parts.append(f'- {key}: {value}')

        provided_info_text = (
            '\n'.join(provided_info_parts)
            if provided_info_parts
            else 'No specific information provided.'
        )

        generated_profile = await agenerate(
            model_name=self.model_name,
            template=self.template_manager.get_template('AGENT_PROFILE'),
            input_values={
                'provided_info': provided_info_text,
            },
            output_parser=PydanticOutputParser(pydantic_object=BaseAgentProfile),
            temperature=temperature,
            **kwargs,
        )

        generated_data = generated_profile.model_dump()
        for key, value in partial_info.items():
            if value is not None and value != '' and value != []:
                generated_data[key] = value

        generated_data['pk'] = unique_id

        final_profile = BaseAgentProfile(**generated_data)

        output_filename = output_file or self.get_default_filename()
        self.save_to_jsonl(final_profile, output_filename)

        return final_profile

    @validate_call
    async def polish_profile(
        self,
        existing_profile: BaseAgentProfile,
        improvement_request: str,
        output_file: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> BaseAgentProfile:
        """
        Polish/improve an existing agent profile based on user request

        Args:
            existing_profile: The agent profile to be improved
            improvement_request: User's specific improvement requirements
            output_file: Output file name (optional)
            temperature: LLM temperature for generation
            **kwargs: Additional parameters

        Returns:
            Improved BaseAgentProfile object
        """
        current_profile_data = existing_profile.model_dump()
        current_profile_text = 'Current agent profile:\n'
        for key, value in current_profile_data.items():
            if key != 'pk' and value is not None and value != '' and value != []:
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                current_profile_text += f'- {key}: {value}\n'

        polish_input = (
            f'{current_profile_text}\nImprovement request: {improvement_request}'
        )

        polished_profile = await agenerate(
            model_name=self.model_name,
            template=self.template_manager.get_template('AGENT_PROFILE'),
            input_values={
                'provided_info': polish_input,
            },
            output_parser=PydanticOutputParser(pydantic_object=BaseAgentProfile),
            temperature=temperature,
            **kwargs,
        )

        polished_data = polished_profile.model_dump()
        polished_data['pk'] = existing_profile.pk

        final_profile = BaseAgentProfile(**polished_data)

        if output_file:
            self.save_to_jsonl(final_profile, output_file)

        return final_profile
