import abc
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, validate_call

from tiny_chat.messages import AgentAction, Message, ScriptEnvironmentResponse

from .dimensions import SotopiaDimensions

log = logging.getLogger('evaluators')

T_eval_dim = TypeVar('T_eval_dim', bound=BaseModel)


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError


class RuleBasedTerminatedEvaluator(Evaluator):
    def __init__(
        self,
        max_turn_number: int = 20,
        max_stale_turn: int = 2,
        leave_detector: Callable[[list[tuple[str, Message]]], bool] | None = None,
    ) -> None:
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn
        self.leave_detector = leave_detector

    @validate_call
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number
        # Rule 2: If one of the players leaves, terminate the conversation (default)
        if self.leave_detector is not None:
            someone_leaving = bool(self.leave_detector(messages))
        else:
            someone_leaving = False
            for source, msg in messages[::-1]:
                if source == 'Environment':
                    continue
                if isinstance(msg, AgentAction) and msg.action_type == 'leave':
                    someone_leaving = True
                    break
        # Rule 3: If the conversation is stale for too long, terminate the conversation
        stale_count = 0
        for message in messages[::-1]:
            if message[0] == 'Environment':
                continue
            assert isinstance(message[1], AgentAction)
            if message[1].action_type == 'none':
                stale_count += 1
            else:
                break
            if stale_count > self.max_stale_turn:
                break
        stale_too_long = stale_count > self.max_stale_turn
        terminated = conversation_too_long or someone_leaving or stale_too_long
        reasons_for_termination = (
            f'{"The conversation is too long; " if conversation_too_long else ""}'
            f'{"Someone is leaving; " if someone_leaving else ""}'
            f'{"The conversation stales for too long; " if stale_too_long else ""}'
        )
        return [
            (
                'environment',
                (('terminated', terminated), reasons_for_termination),
            )
        ]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self(turn_number, messages)


class EpisodeLLMEvaluator(Evaluator, Generic[T_eval_dim]):
    def __init__(
        self,
        model_name: str,
        response_format_class: type[T_eval_dim] | None = None,
    ) -> None:
        self.model_name = model_name
        self.prompt = ''
        self.response_format_class = response_format_class or SotopiaDimensions  # type: ignore

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError(
            'EpisodeLLMEvaluator is not implemented for synchronous evaluation'
        )

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
            log.debug('No conversation history available for evaluation')
            return []

        try:
            from tiny_chat.generator import agenerate
            from tiny_chat.generator.output_parsers import PydanticOutputParser

            # Extract agent names from the conversation
            agent_names = []
            if messages:
                for sender, _ in messages:
                    if sender != 'Environment' and sender not in agent_names:
                        agent_names.append(sender)

            EvaluationClass = EvaluationForMultipleAgents

            # Generate evaluation using agenerate
            response = await agenerate(
                model_name=self.model_name,
                template="""{history}

Based on previous interactions, evaluate how well participants achieve their goals.

IMPORTANT: For each evaluation dimension, provide a tuple with exactly 2 elements:
- First element: a single string containing all reasoning
- Second element: a single integer score

Example format for each dimension:
"believability": ["The agent shows natural behavior and consistency", 8]

Please follow the format:
{format_instructions}
""",
                input_values={'history': history},
                output_parser=PydanticOutputParser(pydantic_object=EvaluationClass),
                temperature=temperature,
            )

            response_list: list[
                tuple[str, tuple[tuple[str, int | float | bool], str]]
            ] = []

            # Convert response to expected format for multiple agents
            for agent_name in agent_names:
                if agent_name in response.agent_evaluations:
                    agent_eval = response.agent_evaluations[agent_name]
                    eval_dict = (
                        agent_eval.dict()
                        if hasattr(agent_eval, 'dict')
                        else agent_eval.__dict__
                    )

                    for dimension, value in eval_dict.items():
                        if isinstance(value, tuple) and len(value) >= 2:
                            # Format: (description, score)
                            score = value[1]
                            reasoning = value[0]
                        else:
                            # Fallback: treat as score
                            score = value
                            reasoning = f'Score for {dimension}'

                        response_list.append(
                            (
                                agent_name,
                                (
                                    (dimension, score),
                                    reasoning,
                                ),
                            )
                        )

            log.debug(f'Generated evaluation for {len(agent_names)} agents')
            return response_list

        except Exception as e:
            print(e)
            log.debug(f'[red] Failed to generate environment response. {e}')
            return []


class EvaluationForMultipleAgents(BaseModel, Generic[T_eval_dim]):
    """Evaluation structure for multiple agents"""

    agent_evaluations: dict[str, T_eval_dim] = Field(
        description='Evaluations for each agent, keyed by agent name'
    )

    class Config:
        title = 'MultiAgentEvaluation'


@validate_call
def _reduce(
    responses_per_reducer: list[tuple[tuple[str, float | int | bool], str]],
) -> tuple[dict[str, float | int | bool], str]:
    responses_dict = defaultdict(list)
    comments_dict: dict[str, str] = defaultdict(str)
    reduced_dict: dict[str, float | int | bool] = {}
    for response, reasoning in responses_per_reducer:
        responses_dict[response[0]].append(response[1])
        comments_dict[response[0]] += reasoning
    scores: list[float | int] = []
    for k, v in responses_dict.items():
        if k == 'terminated':
            assert all(isinstance(x, bool) for x in v)
            reduced_dict[k] = any(v)
        else:
            assert all(isinstance(x, float | int) for x in v)
            reduced_dict[k] = sum(v) / len(v)
            scores.append(reduced_dict[k])
    if len(scores) and 'overall_score' not in responses_dict:
        scores = [x for x in scores if x is not None]
        reduced_dict['overall_score'] = sum(scores) / len(scores)
    comments = '\n'.join([f'{k}: {v}' for k, v in comments_dict.items()])
    return reduced_dict, comments


@validate_call
def unweighted_aggregate_evaluate(
    responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
) -> ScriptEnvironmentResponse:
    """
    Aggregate the responses from the environment

    Args:
        responses (list[tuple[str, tuple[tuple[str, int | bool], str]]]): list of responses from the environment
        Each response is a tuple of (agent_name/environment, (response, reasoning))
    """
    responses_dict: dict[str, list[tuple[tuple[str, int | float | bool], str]]] = (
        defaultdict(list)
    )
    for response in responses:
        responses_dict[response[0]].append(response[1])

    environment_responses: tuple[dict[str, float | int | bool], str] = ({}, '')
    per_agent_responses: dict[str, tuple[dict[str, float | int | bool], str]] = {}
    for k, v in responses_dict.items():
        if k == 'environment':
            environment_responses = _reduce(v)
        else:
            per_agent_responses[k] = _reduce(v)

    comments_parts: list[str] = []
    if environment_responses[1]:
        comments_parts.append(f'Environment comments: {environment_responses[1]}')
    for agent_key, (_, cmt) in per_agent_responses.items():
        if cmt:
            comments_parts.append(f'{agent_key} comments:\n{cmt}')
    comments = '\n'.join(comments_parts)
    if (
        'terminated' in environment_responses[0]
        and environment_responses[0]['terminated']
    ):
        log.debug(f'[green] The conversation is terminated. {responses}')
    per_agent_scores: dict[str, float | tuple[float, dict[str, float]]] = {}
    for agent_key, (score_dict, _) in per_agent_responses.items():
        if 'overall_score' in score_dict:
            per_agent_scores[agent_key] = (score_dict['overall_score'], score_dict)

    return ScriptEnvironmentResponse(
        terminated=(
            environment_responses[0]['terminated']
            if 'terminated' in environment_responses[0]
            else False
        ),
        per_agent_scores=per_agent_scores,
        comments=comments,
    )
