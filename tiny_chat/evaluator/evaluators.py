import abc
import logging
from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel, validate_call

from tiny_chat.messages import AgentAction, Message, ScriptEnvironmentResponse

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
    def __init__(self, max_turn_number: int = 20, max_stale_turn: int = 2) -> None:
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn

    @validate_call
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number
        # Rule 2: If one of the players leaves, terminate the conversation
        p1_leaving = (
            len(messages) > 1
            and isinstance(messages[-2][1], AgentAction)
            and messages[-2][1].action_type == 'leave'
        )
        p2_leaving = (
            bool(len(messages))
            and isinstance(messages[-1][1], AgentAction)
            and messages[-1][1].action_type == 'leave'
        )
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
        terminated = conversation_too_long or p1_leaving or p2_leaving or stale_too_long
        reasons_for_termination = (
            f'{"The conversation is too long; " if conversation_too_long else ""}'
            f'{"Agent 1 is leaving; " if p1_leaving else ""}'
            f'{"Agent 2 is leaving; " if p2_leaving else ""}'
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
        # response_format_class: type[EvaluationForTwoAgents[T_eval_dim]],
    ) -> None:
        self.model_name = model_name
        self.prompt = ''
        # self.response_format_class = response_format_class

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

        try:
            # TODO: Implement actual LLM generation here
            # For now, return empty response
            response_list = []
            return response_list
        except Exception as e:
            print(e)
            log.debug(f'[red] Failed to generate environment response. {e}')
            return []


class TinyChatDimensions(BaseModel):
    """Evaluation dimensions used in Sotopia"""

    overall_score: tuple[str, float] = ('Overall score', 0.0)
    goal_achievement: tuple[str, float] = ('Goal achievement', 0.0)
    social_intelligence: tuple[str, float] = ('Social intelligence', 0.0)
    communication_quality: tuple[str, float] = ('Communication quality', 0.0)


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
        assert response[0] == 'environment' or response[0].startswith('agent')
        responses_dict[response[0]].append(response[1])

    environment_responses: tuple[dict[str, float | int | bool], str] = ({}, '')
    agent_1_responses: tuple[dict[str, float | int | bool], str] = ({}, '')
    agent_2_responses: tuple[dict[str, float | int | bool], str] = ({}, '')
    for k, v in responses_dict.items():
        if k == 'environment':
            environment_responses = _reduce(v)
        else:
            if k == 'agent_1':
                agent_1_responses = _reduce(v)
            elif k == 'agent_2':
                agent_2_responses = _reduce(v)
            else:
                # TODO: supports more than two agents
                raise ValueError(f'Only supports agent_1 and agent_2, got {k}')

    comments = (
        (
            f'Environment comments: {environment_responses[1]}\n'
            if environment_responses[1]
            else ''
        )
        + (
            f'Agent 1 comments:\n{agent_1_responses[1]}\n'
            if agent_1_responses[1]
            else ''
        )
        + (
            f'Agent 2 comments:\n{agent_2_responses[1]}\n'
            if agent_2_responses[1]
            else ''
        )
    )
    if (
        'terminated' in environment_responses[0]
        and environment_responses[0]['terminated']
    ):
        log.debug(f'[green] The conversation is terminated. {responses}')
    return ScriptEnvironmentResponse(
        terminated=(
            environment_responses[0]['terminated']
            if 'terminated' in environment_responses[0]
            else False
        ),
        p1_rate=(
            (
                (
                    agent_1_responses[0]['overall_score']
                    if 'overall_score' in agent_1_responses[0]
                    else 0
                ),
                agent_1_responses[0],
            )
            if agent_1_responses != ({}, '')
            else None
        ),
        p2_rate=(
            (
                (
                    agent_2_responses[0]['overall_score']
                    if 'overall_score' in agent_2_responses[0]
                    else 0
                ),
                agent_2_responses[0],
            )
            if agent_2_responses != ({}, '')
            else None
        ),
        comments=comments,
    )


class EvaluationForTwoAgents:
    def __call__(
        self,
        responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
    ) -> ScriptEnvironmentResponse:
        return unweighted_aggregate_evaluate(responses)


class EvaluationForMultipleAgents:
    @staticmethod
    def _build_agents_rate(
        agent_reduced: dict[str, tuple[dict[str, float | int | bool], str]],
    ) -> dict[str, tuple[float, dict[str, float | int | bool]]]:
        result: dict[str, tuple[float, dict[str, float | int | bool]]] = {}
        for agent_name, (metric_dict, _comments) in agent_reduced.items():
            overall = metric_dict.get('overall_score', 0)
            result[agent_name] = (overall, metric_dict)
        return result

    @validate_call
    def __call__(  # noqa: D401
        self,
        responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
    ) -> ScriptEnvironmentResponse:
        responses_dict: dict[str, list[tuple[tuple[str, int | float | bool], str]]] = (
            defaultdict(list)
        )
        for who, payload in responses:
            responses_dict[who].append(payload)

        env_reduced: tuple[dict[str, float | int | bool], str] = ({}, '')
        agents_reduced: dict[str, tuple[dict[str, float | int | bool], str]] = {}

        for who, payloads in responses_dict.items():
            if who == 'environment':
                env_reduced = _reduce(payloads)
            else:
                agents_reduced[who] = _reduce(payloads)

        terminated = bool(env_reduced[0].get('terminated')) if env_reduced[0] else False

        comment_parts: list[str] = []
        if env_reduced[1]:
            comment_parts.append(f'Environment comments:\n{env_reduced[1]}')
        for agent_name, (_metric, comment) in agents_reduced.items():
            if comment:
                comment_parts.append(f'{agent_name} comments:\n{comment}')
        comments = '\n'.join(comment_parts)

        p1 = agents_reduced.get('agent_1')
        p2 = agents_reduced.get('agent_2')

        return ScriptEnvironmentResponse(
            terminated=terminated,
            agents_rate=self._build_agents_rate(agents_reduced) or None,
            p1_rate=(p1[0].get('overall_score', 0), p1[0]) if p1 else None,
            p2_rate=(p2[0].get('overall_score', 0), p2[0]) if p2 else None,
            comments=comments,
        )
