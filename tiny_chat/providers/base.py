from abc import ABC, abstractmethod
from typing import Any

from litellm import completion

from tiny_chat.config import ModelProviderConfig
from tiny_chat.utils.logger import logger as log

from .generate import (
    agenerate,
    agenerate_action,
    agenerate_goal,
    generate_action,
    generate_goal,
)


class BaseModelProvider(ABC):
    def __init__(self, config: ModelProviderConfig) -> None:
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
        **kwargs: Any,
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

    def _prepare_model_config(
        self, model_name: str
    ) -> tuple[str | None, str | None, str]:
        from .utils import prepare_model_config_from_name

        return prepare_model_config_from_name(model_name)

    def sync_generate_with_parser(
        self,
        prompt: str,
        output_parser: Any,
        temperature: float | None = None,
    ) -> Any:
        model_name = self._get_agenerate_model_name()
        api_base, api_key, eff_model = self._prepare_model_config(model_name)

        messages = [{"role": "user", "content": prompt}]
        try:
            resp = completion(
                model=eff_model,
                messages=messages,
                temperature=(
                    temperature if temperature is not None else self.config.temperature
                ),
                drop_params=True,
                base_url=api_base,
                api_key=api_key,
            )
            text = resp.choices[0].message.content
            assert isinstance(text, str)
            return output_parser.parse(text)
        except Exception as e:
            log.debug(f"[red] sync_generate_with_parser failed: {e}")
            raise

    async def agenerate_action(
        self,
        history: str,
        turn_number: int,
        action_types: list[str],
        agent: str,
        goal: str,
        script_like: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        model_name = self._get_agenerate_model_name()

        return await agenerate_action(
            model_name=model_name,
            history=history,
            turn_number=turn_number,
            action_types=action_types,  # type: ignore[arg-type]
            agent=agent,
            goal=goal,
            temperature=temperature or self.config.temperature,
            script_like=script_like,
            **kwargs,
        )

    def generate_action(
        self,
        history: str,
        turn_number: int,
        action_types: list[str],
        agent: str,
        goal: str,
        script_like: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous version of action generation"""
        model_name = self._get_agenerate_model_name()

        return generate_action(
            model_name=model_name,
            history=history,
            turn_number=turn_number,
            action_types=action_types,  # type: ignore[arg-type]
            agent=agent,
            goal=goal,
            temperature=temperature or self.config.temperature,
            script_like=script_like,
            **kwargs,
        )

    def generate_goal(
        self,
        background: str,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous version of goal generation"""
        model_name = self._get_agenerate_model_name()

        return generate_goal(
            model_name=model_name,
            background=background,
            temperature=temperature or self.config.temperature,
            **kwargs,
        )

    async def agenerate_goal(
        self,
        background: str,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous version of goal generation"""
        model_name = self._get_agenerate_model_name()

        return await agenerate_goal(
            model_name=model_name,
            background=background,
            **kwargs,
        )

    def get_default_temperature(self) -> float:
        return self.config.temperature

    def get_default_max_tokens(self) -> int | None:
        return self.config.max_tokens

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return self.__str__()
