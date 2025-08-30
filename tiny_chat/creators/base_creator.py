import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, validate_call

ProfileType = TypeVar('ProfileType', bound=BaseModel)


class BaseCreator(ABC):
    """Base class for creating profiles with LLM enhancement"""

    def __init__(self, model_name: str = 'gpt-4o-mini', output_dir: str = 'data'):
        """
        Initialize the creator

        Args:
            model_name: Model name for LLM generation
            output_dir: Directory to save created profiles
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_unique_id(self, profile_type: str) -> str:
        """Generate a unique ID for the profile"""
        timestamp = int(datetime.now().timestamp())
        uuid_part = str(uuid.uuid4())[:8]
        return f'{profile_type}_{uuid_part}_{timestamp}'

    @validate_call
    def save_to_jsonl(
        self, profile: BaseModel, filename: str, append: bool = True
    ) -> None:
        """Save profile to JSONL file"""
        file_path = self.output_dir / filename
        mode = 'a' if append else 'w'

        with open(file_path, mode, encoding='utf-8') as f:
            data = profile.model_dump()
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    @abstractmethod
    async def create_profile(
        self, partial_info: dict[str, Any], output_file: str | None = None, **kwargs
    ) -> ProfileType:
        """
        Create a profile with LLM enhancement

        Args:
            partial_info: Partial information provided by user
            output_file: Output file name (optional)
            **kwargs: Additional parameters

        Returns:
            Created profile object
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_filename(self) -> str:
        """Get default output filename for this creator type"""
        raise NotImplementedError

    @abstractmethod
    async def polish_profile(
        self,
        existing_profile: ProfileType,
        improvement_request: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> ProfileType:
        """
        Polish/improve an existing profile based on user request

        Args:
            existing_profile: The profile to be improved
            improvement_request: User's specific improvement requirements
            temperature: LLM temperature for generation
            **kwargs: Additional parameters

        Returns:
            Improved profile object
        """
        raise NotImplementedError
