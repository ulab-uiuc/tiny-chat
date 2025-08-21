from typing import Any

from litellm import acompletion

from ..config import ModelProviderConfig
from .base import BaseModelProvider, ModelResponse


class CustomProvider(BaseModelProvider):
    """Custom model provider for OpenAI-compatible APIs"""

    def __init__(self, config: ModelProviderConfig):
        super().__init__(config)
        if config.type != 'custom':
            raise ValueError(f'Expected custom provider, got {config.type}')

        if not config.api_base:
            raise ValueError('Custom provider requires api_base configuration')

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate text response from prompt"""
        messages = [{'role': 'user', 'content': prompt}]
        return await self.generate_chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate structured response conforming to schema"""
        messages = [{'role': 'user', 'content': prompt}]

        # Prepare parameters for custom API
        params = {
            'model': self.config.name,
            'messages': messages,
            'temperature': temperature or self.get_default_temperature(),
            'base_url': self.config.api_base,
            'drop_params': True,
        }

        if max_tokens or self.get_default_max_tokens():
            params['max_tokens'] = max_tokens or self.get_default_max_tokens()

        if self.config.api_key:
            params['api_key'] = self.config.api_key
        else:
            # Some custom APIs might not require auth
            params['api_key'] = 'EMPTY'

        # Try to use structured output if supported
        try:
            params['response_format'] = schema
        except Exception:
            # If structured output not supported, continue without it
            pass

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await acompletion(**params)

            return ModelResponse(
                content=response.choices[0].message.content or '',
                usage=response.usage.dict() if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={'provider': 'custom', 'api_base': self.config.api_base},
            )
        except Exception as e:
            raise RuntimeError(f'Custom provider generation failed: {e}') from e

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate response from chat messages"""
        # Prepare parameters for custom API
        params = {
            'model': self.config.name,
            'messages': messages,
            'temperature': temperature or self.get_default_temperature(),
            'base_url': self.config.api_base,
            'drop_params': True,
        }

        if max_tokens or self.get_default_max_tokens():
            params['max_tokens'] = max_tokens or self.get_default_max_tokens()

        if self.config.api_key:
            params['api_key'] = self.config.api_key
        else:
            # Some custom APIs might not require auth
            params['api_key'] = 'EMPTY'

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await acompletion(**params)

            return ModelResponse(
                content=response.choices[0].message.content or '',
                usage=response.usage.dict() if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={'provider': 'custom', 'api_base': self.config.api_base},
            )
        except Exception as e:
            raise RuntimeError(f'Custom provider generation failed: {e}') from e

    async def check_health(self) -> bool:
        """Check if custom API is accessible"""
        try:
            # Send a minimal request to check connectivity
            await self.generate(prompt='Hello', max_tokens=1, temperature=0.0)
            return True
        except Exception:
            return False
