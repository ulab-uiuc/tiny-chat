from typing import Any, Literal

from tiny_chat.agents import LLMAgent
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.evaluator import (
    EpisodeLLMEvaluator,
    Evaluator,
    RuleBasedTerminatedEvaluator,
    TinyChatDimensions,
)
from tiny_chat.generator import agenerate_goal
from tiny_chat.messages import UnifiedChatBackground


class TinyChatServer:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    async def run_conversation(
        self,
        agent_configs: list[dict[str, Any]],
        background: UnifiedChatBackground | None = None,
        max_turns: int = 20,
        enable_evaluation: bool = True,
        action_order: Literal[
            'simultaneous', 'round-robin', 'sequential', 'random'
        ] = 'simultaneous',
        scenario: str | None = None,
    ) -> None:
        evaluators: list[Evaluator] = [
            RuleBasedTerminatedEvaluator(max_turn_number=max_turns)
        ]
        terminal_evaluators: list[Evaluator] = []

        if enable_evaluation:
            terminal_evaluators.append(
                EpisodeLLMEvaluator[TinyChatDimensions](model_name='gpt-4o-mini')
            )

        env = TinyChatEnvironment(
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
            action_order=action_order,
            max_turns=max_turns,
        )

        agents = await self._create_agents(agent_configs, background)

        reset_options = {}
        if scenario:
            reset_options['scenario'] = scenario
        elif background:
            reset_options['scenario'] = background.to_natural_language()

        env.reset(agents=agents, options=reset_options if reset_options else None)

        num_agents = len(agents)
        print(f'=== {num_agents}-Agent Conversation Starting ===')
        if background:
            print(background.to_natural_language())
        elif scenario:
            print(scenario)
        print('=' * 50)

        await self._run_conversation_loop(env, agents)

        print(f'\n=== {len(agents)}-Agent Conversation Ended ===')
        print(f'Total turns: {env.get_turn_number()}')

        if enable_evaluation and terminal_evaluators:
            print('\n=== Running Episode Evaluation ===')
            evaluation_results = await env.evaluate_episode()
            self._print_evaluation_results(evaluation_results)

        print('\n=== Full Conversation Summary ===')
        print(env.get_conversation_summary())

    async def _create_agents(
        self, agent_configs: list[dict[str, Any]], background: Any | None = None
    ) -> dict[str, Any]:
        agents = {}

        for config in agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']

            if agent_type == 'llm':
                agent = LLMAgent(
                    agent_name=name,
                )
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            if 'goal' in config:
                agent.goal = config['goal']
            elif background and agent_type == 'llm':
                agent.goal = await agenerate_goal(
                    model_name='gpt-4o-mini',
                    background=background.to_natural_language(),
                )

            agents[name] = agent

        return agents

    async def _run_conversation_loop(self, env: Any, agents: dict[str, Any]) -> None:
        while not env.is_terminated():
            print(f'\n--- Turn {env.get_turn_number()} ---')

            actions = {}
            for name, agent in agents.items():
                observation = env.get_observation(name)
                action = await agent.act(observation)
                actions[name] = action

                print(f'{name}: {action.to_natural_language()}')

            await env.astep(actions)

    def _print_evaluation_results(self, results: dict[str, Any]) -> None:
        if 'message' in results:
            print(f'Evaluation: {results["message"]}')
            return

        print(f'Conversation Terminated: {results.get("terminated", False)}')

        agent_scores = {}
        agent_details = {}

        for key, value in results.items():
            if key.endswith('_score'):
                agent_name = key[:-6]
                agent_scores[agent_name] = value
            elif key.endswith('_details'):
                agent_name = key[:-8]
                agent_details[agent_name] = value

        for agent_name, score in agent_scores.items():
            print(f'{agent_name} Overall Score: {score:.2f}')

        for agent_name, details in agent_details.items():
            print(f'\n{agent_name} Detailed Scores:')
            for dimension, score in details.items():
                if dimension != 'overall_score':
                    print(f'  {dimension}: {score}')

        if results.get('comments'):
            print(f'\nEvaluation Comments:\n{results["comments"]}')
