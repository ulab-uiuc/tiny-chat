from typing import Any, Literal

from tiny_chat.agents import LLMAgent, TwoAgentManager, MultiAgentManager, LLMGenerator
from tiny_chat.envs import MultiAgentTinyChatEnvironment, TwoAgentTinyChatEnvironment
from tiny_chat.evaluator import EpisodeLLMEvaluator, RuleBasedTerminatedEvaluator
from tiny_chat.generator import agenerate_goal
from tiny_chat.messages import MultiAgentChatBackground, TwoAgentChatBackground


class TinyChatServer:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    async def two_agent_run_conversation(
        self,
        agent_configs: list[dict[str, Any]],
        background: TwoAgentChatBackground | None = None,
        general_settings: dict[str, Any] | None = None,
        enable_evaluation: bool = True,
    ):
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=general_settings['max_turns'])]
        terminal_evaluators = []

        if enable_evaluation:
            terminal_evaluators.append(EpisodeLLMEvaluator(model_name='gpt-4o-mini'))

        env = TwoAgentTinyChatEnvironment(
            background_class=TwoAgentChatBackground,
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
        )
        env.max_turns = general_settings['max_turns']
        self.general_settings = general_settings
        agents = {}
        agent_manager = TwoAgentManager(
            agent_configs=agent_configs,
            background=background,
            api_key=self.api_key,
            general_settings=self.general_settings,
        )
        await agent_manager.initialize_agents()
        agents = agent_manager.agents

        env.reset(agents=agents)

        print('=== Conversation Starting ===')
        if background:
            print(background.to_natural_language())
        print('=' * 50)
        if self.general_settings['use_same_model']:
            await self._run_conversation_loop(env, agents, agent_manager.shared_llm_generator)
        else:
            await self._run_conversation_loop(env, agents, None)
        

        print('\n=== Conversation Ended ===')
        print(f'Total turns: {env.get_turn_number()}')

        # if enable_evaluation and terminal_evaluators:
        #     print('\n=== Running Episode Evaluation ===')
        #     evaluation_results = await env.evaluate_episode()
        #     self._print_evaluation_results(evaluation_results)

        # print('\n=== Full Conversation Summary ===')
        # print(env.get_conversation_summary())

    async def multi_agent_run_conversation(
        self,
        agent_configs: list[dict[str, Any]],
        background: MultiAgentChatBackground | None = None,
        max_turns: int = 20,
        enable_evaluation: bool = True,
        action_order: Literal[
            'simultaneous', 'round-robin', 'sequential', 'random'
        ] = 'sequential',
    ) -> None:
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=max_turns)]
        terminal_evaluators = []

        if enable_evaluation:
            terminal_evaluators.append(EpisodeLLMEvaluator(model_name='gpt-4o-mini'))

        env = MultiAgentTinyChatEnvironment(
            background_class=MultiAgentChatBackground,
            evaluators=evaluators,
            action_order=action_order,
            terminal_evaluators=terminal_evaluators,
            max_turns=max_turns,
        )

        agent_manager = MultiAgentManager(
            agent_configs=agent_configs,
            background=background,
            api_key=self.api_key,
            general_settings={'max_turns': max_turns},
        )
        await agent_manager.initialize_agents()
        agents = agent_manager.agents

        env.reset(agents=agents)

        print('=== Multi-Agent Conversation Starting ===')
        if background:
            print(background.to_natural_language())
        print('=' * 50)

        await self._run_conversation_loop(env, agents)

        print('\n=== Multi-Agent Conversation Ended ===')
        print(f'Total turns: {env.get_turn_number()}')

        if enable_evaluation and terminal_evaluators:
            print('\n=== Running Episode Evaluation ===')
            evaluation_results = await env.evaluate_episode()
            self._print_evaluation_results(evaluation_results)

        print('\n=== Full Conversation Summary ===')
        print(env.get_conversation_summary())


    async def _run_conversation_loop(self, env: Any, agents: dict[str, Any], shared_llm_generator: LLMGenerator | None = None) -> None:
        while not env.is_terminated():
            print(f'\n--- Turn {env.get_turn_number()} ---')
            if self.general_settings['use_same_model']:
                
                actions = {}
                for name, agent in agents.items():
                    observation = env.get_observation(name)
                    action = await agent.act(observation, shared_llm_generator)
                    actions[name] = action

                    print(f'{name}: {action.to_natural_language()}')

                await env.astep(actions)
            else:
                actions = {}
                for name, agent in agents.items():
                    observation = env.get_observation(name)
                    action = await agent.act(observation, agent.llm_generator)
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
