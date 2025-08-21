from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ..config import ModelProviderConfig


class ModelResponse(BaseModel):
    """Response from model generation"""

    content: str
    usage: dict[str, int] | None = None
    model: str
    finish_reason: str | None = None
    metadata: dict[str, Any] = {}


class BaseModelProvider(ABC):
    """Abstract base class for model providers"""

    def __init__(self, config: ModelProviderConfig):
        self.config = config
        self.name = config.name
        self.type = config.type

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate text response from prompt"""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate structured response conforming to schema"""
        pass

    @abstractmethod
    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate response from chat messages"""
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the provider is healthy and accessible"""
        pass

    def get_default_temperature(self) -> float:
        """Get default temperature for this provider"""
        return self.config.temperature

    def get_default_max_tokens(self) -> int | None:
        """Get default max tokens for this provider"""
        return self.config.max_tokens

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    def __repr__(self) -> str:
        return self.__str__()
