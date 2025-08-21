from abc import ABC, abstractmethod
from typing import Any

from tiny_chat.generator import agenerate, agenerate_action, agenerate_goal

from ..config import ModelProviderConfig


class BaseModelProvider(ABC):
    def __init__(self, config: ModelProviderConfig):
        self.config = config
        self.name = config.name
        self.type = config.type

    @abstractmethod
    def _get_agenerate_model_name(self) -> str:
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        pass

    async def agenerate(
        self,
        template: str,
        input_values: dict[str, str],
        output_parser: Any,
        temperature: float | None = None,
        structured_output: bool = False,
        **kwargs,
    ) -> Any:
        model_name = self._get_agenerate_model_name()

        return await agenerate(
            model_name=model_name,
            template=template,
            input_values=input_values,
            output_parser=output_parser,
            temperature=temperature or self.config.temperature,
            structured_output=structured_output,
            **kwargs,
        )

    async def agenerate_action(
        self,
        history: str,
        turn_number: int,
        action_types: list[str],
        agent: str,
        goal: str,
        script_like: bool = False,
        temperature: float | None = None,
        **kwargs,
    ) -> Any:
        model_name = self._get_agenerate_model_name()

        return await agenerate_action(
            model_name=model_name,
            history=history,
            turn_number=turn_number,
            action_types=action_types,
            agent=agent,
            goal=goal,
            temperature=temperature or self.config.temperature,
            script_like=script_like,
            **kwargs,
        )

    async def agenerate_goal(
        self,
        background: str,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        model_name = self._get_agenerate_model_name()

        return await agenerate_goal(
            model_name=model_name,
            background=background,
            temperature=temperature or self.config.temperature,
            **kwargs,
        )

    def get_default_temperature(self) -> float:
        return self.config.temperature

    def get_default_max_tokens(self) -> int | None:
        return self.config.max_tokens

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    def __repr__(self) -> str:
        return self.__str__()
