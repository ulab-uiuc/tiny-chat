from abc import ABC, abstractmethod
from typing import Any

from litellm import completion

from tiny_chat.generator import (
    agenerate,
    agenerate_action,
    agenerate_goal,
    generate_action,
    generate_goal,
)
from tiny_chat.generator.output_parsers import PydanticOutputParser
from tiny_chat.utils.logger import logger as log

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

    def generate_action(
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
        """Synchronous version of action generation"""
        model_name = self._get_agenerate_model_name()

        return generate_action(
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

    def generate_goal(
        self,
        background: str,
        temperature: float | None = None,
        **kwargs,
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
        **kwargs,
    ) -> str:
        """Asynchronous version of goal generation"""
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

    def generate_evaluation(
        self,
        history: str,
        evaluation_class: type,
        temperature: float | None = None,
    ) -> Any:
        """
        Synchronous LLM-judge generation.
        This mirrors EpisodeLLMEvaluator.__acall__ but uses a blocking LiteLLM 'completion(...)'.
        """
        model_name = self._get_agenerate_model_name()

        output_parser = PydanticOutputParser(pydantic_object=evaluation_class)
        format_instructions = output_parser.get_format_instructions()

        prompt = f"""{history}

Based on previous interactions, evaluate how well participants achieve their goals.

IMPORTANT: For each evaluation dimension, provide a tuple with exactly 2 elements:
- First element: a single string containing all reasoning
- Second element: a single integer score

Example format for each dimension:
"believability": ["The agent shows natural behavior and consistency", 8]

Please follow the format:
{format_instructions}
"""

        try:
            from tiny_chat.generator import _prepare_provider_config  # type: ignore

            api_base, api_key, eff_model = _prepare_provider_config(model_name)
        except Exception:
            api_base, api_key, eff_model = (None, None, model_name)

        messages = [{'role': 'user', 'content': prompt}]
        try:
            resp = completion(
                model=eff_model,
                messages=messages,
                temperature=temperature
                if temperature is not None
                else self.config.temperature,
                drop_params=True,
                base_url=api_base,
                api_key=api_key,
            )
            text = resp.choices[0].message.content
            assert isinstance(text, str)
            return output_parser.parse(text)
        except Exception as e:
            log.debug(f'[red] generate_evaluation failed to parse: {e}')
            raise
