import logging
from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, validate_call

from tiny_chat.messages import ScriptEnvironmentResponse

logger = logging.getLogger(__name__)

T_eval_dim = TypeVar('T_eval_dim', bound=BaseModel)


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
        logger.debug(f'[green] The conversation is terminated. {responses}')
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
