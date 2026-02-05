from typing import Any

from pydantic import validate_call

from tiny_chat.generator.generate import agenerate
from tiny_chat.generator.output_parsers import PydanticOutputParser
from tiny_chat.profiles.relationship_profile import BaseRelationshipProfile
from tiny_chat.utils.template import TemplateManager

from .base_creator import BaseCreator


class RelationshipCreator(BaseCreator):
    """Creator for relationship profiles with LLM enhancement"""

    def __init__(self, model_name: str = 'gpt-4o-mini', output_dir: str = 'data'):
        super().__init__(model_name, output_dir)
        self.template_manager = TemplateManager()

    def get_default_filename(self) -> str:
        """Get default output filename"""
        return 'relationship_profiles.jsonl'

    @validate_call
    async def create_profile(
        self,
        partial_info: dict[str, Any],
        output_file: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> BaseRelationshipProfile:
        """
        Create a relationship profile with LLM enhancement

        Args:
            partial_info: Partial information provided by user
            output_file: Output file name (defaults to relationship_profiles.jsonl)
            temperature: LLM temperature for generation
            **kwargs: Additional parameters

        Returns:
            Created BaseRelationshipProfile object
        """
        # Generate unique ID
        unique_id = self.generate_unique_id('rel')

        # Prepare provided information text
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

        # Generate complete profile using LLM
        generated_profile = await agenerate(
            model_name=self.model_name,
            template=self.template_manager.get_template('RELATIONSHIP_PROFILE_CREATOR'),
            input_values={
                'provided_info': provided_info_text,
            },
            output_parser=PydanticOutputParser(pydantic_object=BaseRelationshipProfile),
            temperature=temperature,
            **kwargs,
        )

        # Override with user-provided values
        generated_data = generated_profile.model_dump()
        for key, value in partial_info.items():
            if value is not None and value != '' and value != []:
                generated_data[key] = value

        # Set the unique ID
        generated_data['pk'] = unique_id

        # Ensure agent_ids is a list (not set) for JSON serialization
        if 'agent_ids' in generated_data and isinstance(
            generated_data['agent_ids'], set
        ):
            generated_data['agent_ids'] = list(generated_data['agent_ids'])

        # Create final profile object
        final_profile = BaseRelationshipProfile(**generated_data)

        # Save to file
        output_filename = output_file or self.get_default_filename()
        self.save_to_jsonl(final_profile, output_filename)

        return final_profile

    @validate_call
    async def polish_profile(
        self,
        existing_profile: BaseRelationshipProfile,
        improvement_request: str,
        output_file: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> BaseRelationshipProfile:
        """
        Polish/improve an existing relationship profile based on user request

        Args:
            existing_profile: The relationship profile to be improved
            improvement_request: User's specific improvement requirements
            output_file: Output file name (optional)
            temperature: LLM temperature for generation
            **kwargs: Additional parameters

        Returns:
            Improved BaseRelationshipProfile object
        """
        # Prepare current profile information
        current_profile_data = existing_profile.model_dump()
        current_profile_text = 'Current relationship profile:\n'
        for key, value in current_profile_data.items():
            if key != 'pk' and value is not None and value != '' and value != []:
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                current_profile_text += f'- {key}: {value}\n'

        # Create polish request text
        polish_input = (
            f'{current_profile_text}\nImprovement request: {improvement_request}'
        )

        # Generate polished profile using LLM
        polished_profile = await agenerate(
            model_name=self.model_name,
            template=self.template_manager.get_template('RELATIONSHIP_PROFILE_CREATOR'),
            input_values={
                'provided_info': polish_input,
            },
            output_parser=PydanticOutputParser(pydantic_object=BaseRelationshipProfile),
            temperature=temperature,
            **kwargs,
        )

        # Keep the original pk
        polished_data = polished_profile.model_dump()
        polished_data['pk'] = existing_profile.pk

        # Ensure agent_ids is a list (not set) for JSON serialization
        if 'agent_ids' in polished_data and isinstance(polished_data['agent_ids'], set):
            polished_data['agent_ids'] = list(polished_data['agent_ids'])

        # Create final polished profile object
        final_profile = BaseRelationshipProfile(**polished_data)

        # Save to file if requested
        if output_file:
            self.save_to_jsonl(final_profile, output_file)

        return final_profile
