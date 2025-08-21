from typing import Any

from litellm import acompletion

from ..config import ModelProviderConfig
from .base import BaseModelProvider, ModelResponse


class LiteLLMProvider(BaseModelProvider):
    """Universal LiteLLM provider supporting OpenAI, Anthropic, Together, vLLM, Ollama, and custom endpoints"""

    def __init__(self, config: ModelProviderConfig):
        super().__init__(config)
        self.supported_types = {
            "openai",
            "anthropic",
            "together",
            "vllm",
            "ollama",
            "bedrock",
            "azure",
            "palm",
            "cohere",
            "replicate",
            "litellm",
            "custom",
        }

        self.full_model_name = self._prepare_model_name()

    def _prepare_model_name(self) -> str:
        """Prepare the full model name with provider prefix if needed"""
        model_name = self.config.name
        provider_type = self.config.type

        if provider_type == "anthropic" and not model_name.startswith("claude"):
            return f"anthropic/{model_name}"
        elif provider_type == "together" and not model_name.startswith("together/"):
            return f"together/{model_name}"
        elif provider_type == "vllm" and not model_name.startswith("openai/"):
            return f"openai/{model_name}"  # vLLM uses OpenAI-compatible format
        elif provider_type == "ollama" and not model_name.startswith("ollama/"):
            return f"ollama/{model_name}"
        elif provider_type == "bedrock" and not model_name.startswith("bedrock/"):
            return f"bedrock/{model_name}"
        elif provider_type == "azure" and not model_name.startswith("azure/"):
            return f"azure/{model_name}"
        elif provider_type == "cohere" and not model_name.startswith("cohere/"):
            return f"cohere/{model_name}"
        elif provider_type == "replicate" and not model_name.startswith("replicate/"):
            return f"replicate/{model_name}"
        elif provider_type == "custom":
            return model_name
        else:
            return model_name

    def _prepare_params(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare parameters for LiteLLM completion"""
        params = {
            "model": self.full_model_name,
            "messages": messages,
            "temperature": temperature or self.get_default_temperature(),
            "drop_params": True,
        }

        if max_tokens or self.get_default_max_tokens():
            params["max_tokens"] = max_tokens or self.get_default_max_tokens()

        if self.config.api_base:
            params["base_url"] = self.config.api_base

        if self.config.api_key:
            params["api_key"] = self.config.api_key
        elif self.config.type == "custom":
            params["api_key"] = "EMPTY"

        if self.config.timeout:
            params["timeout"] = self.config.timeout

        params.update(kwargs)

        return params

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate text response from prompt"""
        messages = [{"role": "user", "content": prompt}]
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
        messages = [{"role": "user", "content": prompt}]

        params = self._prepare_params(
            messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        try:
            params["response_format"] = schema
        except Exception:
            pass

        try:
            response = await acompletion(**params)

            return ModelResponse(
                content=response.choices[0].message.content or "",
                usage=response.usage.dict() if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "provider": self.config.type,
                    "api_base": self.config.api_base,
                    "structured_output": True,
                },
            )
        except Exception as e:
            raise RuntimeError(f"LiteLLM generation failed: {e}") from e

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate response from chat messages"""
        params = self._prepare_params(
            messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        try:
            response = await acompletion(**params)

            return ModelResponse(
                content=response.choices[0].message.content or "",
                usage=response.usage.dict() if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "provider": self.config.type,
                    "api_base": self.config.api_base,
                },
            )
        except Exception as e:
            raise RuntimeError(f"LiteLLM generation failed: {e}") from e

    async def check_health(self) -> bool:
        """Check if the model provider is accessible"""
        try:
            await self.generate(prompt="Hello", max_tokens=1, temperature=0.0)
            return True
        except Exception:
            return False
