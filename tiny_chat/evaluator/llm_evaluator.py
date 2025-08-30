import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, validate_call

from tiny_chat.messages import Message

from .base import BaseEvaluator
from .dimensions import SotopiaDimensions

if TYPE_CHECKING:
    from tiny_chat.providers import BaseModelProvider

logger = logging.getLogger(__name__)

T_eval_dim = TypeVar('T_eval_dim', bound=BaseModel)


def _create_default_evaluator_model_provider() -> 'BaseModelProvider':
    """Create a default model provider for evaluation using gpt-4o-mini"""
    from tiny_chat.config import ModelProviderConfig
    from tiny_chat.providers import ModelProviderFactory

    default_config = ModelProviderConfig(
        name='gpt-4o-mini',
        type='openai',
        temperature=0.0,  # Use 0.0 for evaluations for consistency
    )
    return ModelProviderFactory.create_provider(default_config)


class LLMEvaluator(BaseEvaluator, Generic[T_eval_dim]):
    def __init__(
        self,
        config: dict[str, Any] = None,
        model_provider: 'BaseModelProvider | None' = None,
        response_format_class: type[T_eval_dim] | None = None,
    ) -> None:
        if config is not None:
            super().__init__(config)
            self._model_provider = (
                config.get('model_provider', model_provider)
                or _create_default_evaluator_model_provider()
            )
            self.response_format_class = (
                config.get('response_format_class', response_format_class)
                or SotopiaDimensions  # type: ignore
            )
        else:
            super().__init__({})
            self._model_provider = (
                model_provider or _create_default_evaluator_model_provider()
            )
            self.response_format_class = response_format_class or SotopiaDimensions  # type: ignore
        self.prompt = ''

    @property
    def evaluator_type(self) -> str:
        return 'llm'

    def _build_evaluation_prompt(self, history: str) -> str:
        from tiny_chat.providers.output_parsers import PydanticOutputParser

        from .utils import EvaluationForMultipleAgents

        EvaluationClass = EvaluationForMultipleAgents[self.response_format_class]
        output_parser = PydanticOutputParser(pydantic_object=EvaluationClass)
        format_instructions = output_parser.get_format_instructions()

        return f"""{history}

    Based on previous interactions, evaluate how well participants achieve their goals.

    IMPORTANT: For each evaluation dimension, provide a tuple with exactly 2 elements:
    - First element: a single string containing all reasoning
    - Second element: a single integer score

    Example format for each dimension:
    "believability": ["The agent shows natural behavior and consistency", 8]

    Please follow the format:
    {format_instructions}
    """

    def _process_evaluation_response(
        self, response: any, agent_names: list[str]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        response_list: list[tuple[str, tuple[tuple[str, int | float | bool], str]]] = []

        for agent_name in agent_names:
            if (
                hasattr(response, 'agent_evaluations')
                and agent_name in response.agent_evaluations
            ):
                agent_eval = response.agent_evaluations[agent_name]
                eval_dict = (
                    agent_eval.dict()
                    if hasattr(agent_eval, 'dict')
                    else agent_eval.__dict__
                )
                for dimension, value in eval_dict.items():
                    if isinstance(value, tuple) and len(value) >= 2:
                        reasoning, score = value[0], value[1]
                    else:
                        score = value
                        reasoning = f'Score for {dimension}'
                    response_list.append((agent_name, ((dimension, score), reasoning)))
        return response_list

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        history_parts: list[str] = []
        for sender, msg in messages:
            if 'did nothing' in msg.to_natural_language():
                continue
            if sender == 'Environment':
                history_parts.append(msg.to_natural_language())
            else:
                history_parts.append(f'{sender} {msg.to_natural_language()}')
        history = '\n'.join(history_parts).strip()

        if not history:
            return []
        agent_names: list[str] = []
        for sender, _ in messages:
            if sender != 'Environment' and sender not in agent_names:
                agent_names.append(sender)

        from tiny_chat.providers.output_parsers import PydanticOutputParser

        from .utils import EvaluationForMultipleAgents

        EvaluationClass = EvaluationForMultipleAgents[self.response_format_class]

        try:
            prompt = self._build_evaluation_prompt(history)

            output_parser = PydanticOutputParser(pydantic_object=EvaluationClass)

            response = self._model_provider.sync_generate_with_parser(
                prompt=prompt,
                output_parser=output_parser,
                temperature=0.0,
            )

            response_list = self._process_evaluation_response(response, agent_names)
            return response_list

        except Exception as e:
            logger.debug(f'[red] Sync LLM evaluation failed: {e}')
            return []

    @validate_call
    async def __acall__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]] | None,
        history: str = '',
        temperature: float = 0.0,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # filter did nothing
        if not history and messages:
            messages_filtered = [
                (x, y)
                for x, y in messages
                if 'did nothing' not in y.to_natural_language()
            ]
            history = '\n'.join(
                [
                    (
                        f'{x} {y.to_natural_language()}'
                        if x != 'Environment'
                        else y.to_natural_language()
                    )
                    for x, y in messages_filtered
                ]
            )

        if not history.strip():
            logger.debug('No conversation history available for evaluation')
            return []

        try:
            agent_names = []
            if messages:
                for sender, _ in messages:
                    if sender != 'Environment' and sender not in agent_names:
                        agent_names.append(sender)
            prompt = self._build_evaluation_prompt(history)

            from tiny_chat.providers.output_parsers import PydanticOutputParser

            from .utils import EvaluationForMultipleAgents

            EvaluationClass = EvaluationForMultipleAgents[self.response_format_class]
            output_parser = PydanticOutputParser(pydantic_object=EvaluationClass)

            if hasattr(self._model_provider, 'agenerate_evaluation'):
                response = await self._model_provider.agenerate_evaluation(
                    history=history,
                    evaluation_class=EvaluationClass,
                    temperature=temperature,
                )
            else:
                from tiny_chat.providers.generate import agenerate

                effective_model_name = self._model_provider._get_agenerate_model_name()
                response = await agenerate(
                    model_name=effective_model_name,
                    template=prompt,
                    input_values={'history': history},
                    output_parser=output_parser,
                    temperature=temperature,
                )

            return self._process_evaluation_response(response, agent_names)

        except Exception as e:
            logger.debug(f'[red] Failed to generate environment response. {e}')
            return []

    def get_terminal_evaluator(self):
        return self
