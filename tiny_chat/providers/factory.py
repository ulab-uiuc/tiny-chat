from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from tiny_chat.config import ModelProviderConfig

from .base import BaseModelProvider
from .litellm_provider import LiteLLMProvider
from .workflow_provider import WorkflowProvider


class ModelProviderFactory:
    """Factory for creating model provider instances"""

    _providers: dict[str, type[BaseModelProvider]] = {
        # LiteLLM provider supports many types
        'openai': LiteLLMProvider,
        'anthropic': LiteLLMProvider,
        'together': LiteLLMProvider,
        'vllm': LiteLLMProvider,
        'ollama': LiteLLMProvider,
        'bedrock': LiteLLMProvider,
        'azure': LiteLLMProvider,
        'palm': LiteLLMProvider,
        'cohere': LiteLLMProvider,
        'replicate': LiteLLMProvider,
        'litellm': LiteLLMProvider,
        # Custom endpoints now supported by LiteLLM provider
        'custom': LiteLLMProvider,
        # Workflow provider for extensible use cases
        'workflow': WorkflowProvider,
    }

    @classmethod
    def create_provider(cls, config: ModelProviderConfig) -> BaseModelProvider:
        """Create a model provider instance from configuration"""
        provider_class = cls._providers.get(config.type)

        if provider_class is None:
            available_types = ', '.join(cls._providers.keys())
            raise ValueError(
                f'Unknown provider type: {config.type}. '
                f'Available types: {available_types}'
            )

        return provider_class(config)

    @classmethod
    def register_provider(
        cls, provider_type: str, provider_class: type[BaseModelProvider]
    ) -> None:
        """Register a new provider type"""
        cls._providers[provider_type] = provider_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available provider types"""
        return list(cls._providers.keys())

    @classmethod
    def is_supported(cls, provider_type: str) -> bool:
        """Check if a provider type is supported"""
        return provider_type in cls._providers
