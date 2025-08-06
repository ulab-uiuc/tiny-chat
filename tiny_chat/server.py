"""
Simple Chat Server - Run multi-agent conversations
"""

import asyncio
from typing import Literal

from .envs import TwoAgentTinyChatEnvironment, MultiAgentTinyChatEnvironment
from .evaluator import (
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from .messages import TwoAgentChatBackground, MultiAgentChatBackground
from .generator import agenerate_goal

    async def two_agent_run_conversation(
        self,
        agent_configs: list[dict],
        background: TwoAgentChatBackground | None = None,
        max_turns: int = 20,
        enable_evaluation: bool = True,
    ):
        """Run a conversation with specified agents"""

        # Create evaluators
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=max_turns)]
        terminal_evaluators = []

        if enable_evaluation:
            terminal_evaluators.append(EpisodeLLMEvaluator(model_name='gpt-4o-mini'))

        # Create environment
        env = TwoAgentTinyChatEnvironment(
            background_class=TwoAgentChatBackground,
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
        )
        env.max_turns = max_turns

        # Create and add agents
        agents = {}
        for config in agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']
            agent_number = config.get('agent_number', 1)

            if agent_type == 'llm':
                model = config.get('model', 'gpt-4o-mini')
                agent = LLMAgent(
                    agent_name=name,
                    agent_number=agent_number,
                    model=model,
                    api_key=self.api_key,
                )
            # elif agent_type == 'human':
            #     agent = HumanAgent(name=name, agent_number=agent_number)
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            # Set goal if provided
            if 'goal' in config:
                agent.goal = config['goal']
            elif background and agent_type == 'llm':
                # Generate goal from background
                agent.goal = await agenerate_goal(
                    model_name='gpt-4o-mini',
                    background=background.to_natural_language(),
                )

            agents[name] = agent

        # Reset environment with agents
        env.reset(agents=agents)

        print('=== Conversation Starting ===')
        if background:
            print(background.to_natural_language())
        print('=' * 50)

        # Main conversation loop
        while not env.is_terminated():
            print(f'\n--- Turn {env.get_turn_number()} ---')

            # Get actions from all agents
            actions = {}
            for name, agent in agents.items():
                observation = env.get_observation(name)
                action = await agent.act(observation)
                actions[name] = action

                # Print the action
                print(f'{name}: {action.to_natural_language()}')

            # Execute step
            await env.astep(actions)

        print('\n=== Conversation Ended ===')
        print(f'Total turns: {env.get_turn_number()}')

        # Run episode evaluation if enabled
        if enable_evaluation and terminal_evaluators:
            print('\n=== Running Episode Evaluation ===')
            evaluation_results = await env.evaluate_episode()
            self._print_evaluation_results(evaluation_results)

        print('\n=== Full Conversation Summary ===')
        print(env.get_conversation_summary())

    async def multi_agent_run_conversation(
        self,
        agent_configs: list[dict],
        background: MultiAgentChatBackground | None = None,
        max_turns: int = 20,
        enable_evaluation: bool = True,
        action_order: Literal['simultaneous', 'round-robin', 'sequential', 'random'] = 'sequential',
    ):
        """Run a multi-agent conversation with specified agents"""
        
        # Create evaluators
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=max_turns)]
        terminal_evaluators = []

        if enable_evaluation:
            terminal_evaluators.append(EpisodeLLMEvaluator(model_name='gpt-4o-mini'))

        # Create environment
        env = MultiAgentTinyChatEnvironment(
            background_class=MultiAgentChatBackground,
            evaluators=evaluators,
            action_order=action_order,
            terminal_evaluators=terminal_evaluators,
            max_turns=max_turns,
        )

        # Create and add agents
        agents = {}
        for config in agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']
            agent_number = config.get('agent_number', 1)

            if agent_type == 'llm':
                model = config.get('model', 'gpt-4o-mini')
                agent = LLMAgent(
                    agent_name=name,
                    agent_number=agent_number,
                    model=model,
                    api_key=self.api_key,
                )
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            # Set goal if provided
            if 'goal' in config:
                agent.goal = config['goal']
            elif background and agent_type == 'llm':
                # Generate goal from background using existing function
                agent.goal = await agenerate_goal(
                    model_name="gpt-4o-mini",
                    background=background.to_natural_language(),
                )

            agents[name] = agent

        # Reset environment with agents
        env.reset(agents=agents)

        print('=== Multi-Agent Conversation Starting ===')
        if background:
            print(background.to_natural_language())
        print('=' * 50)

        # Main conversation loop
        while not env.is_terminated():
            print(f'\n--- Turn {env.get_turn_number()} ---')

            # Get actions from all agents based on action order
            actions = {}
            for name, agent in agents.items():
                observation = env.get_observation(name)
                action = await agent.act(observation)
                actions[name] = action

                # Print the action
                print(f'{name}: {action.to_natural_language()}')

            # Execute step
            await env.astep(actions)

        print('\n=== Multi-Agent Conversation Ended ===')
        print(f'Total turns: {env.get_turn_number()}')

        # Run episode evaluation if enabled
        if enable_evaluation and terminal_evaluators:
            print('\n=== Running Episode Evaluation ===')
            evaluation_results = await env.evaluate_episode()
            self._print_evaluation_results(evaluation_results)

        print('\n=== Full Conversation Summary ===')
        print(env.get_conversation_summary())
    
    def _print_evaluation_results(self, results: dict):
        """Print evaluation results in a formatted way"""
        if 'message' in results:
            print(f'Evaluation: {results["message"]}')
            return

        print(f'Conversation Terminated: {results.get("terminated", False)}')

        # Print scores for all agents
        agent_scores = {}
        agent_details = {}

        for key, value in results.items():
            if key.endswith('_score'):
                agent_name = key[:-6]  # Remove '_score' suffix
                agent_scores[agent_name] = value
            elif key.endswith('_details'):
                agent_name = key[:-8]  # Remove '_details' suffix
                agent_details[agent_name] = value

        # Print overall scores
        for agent_name, score in agent_scores.items():
            print(f'{agent_name} Overall Score: {score:.2f}')

        # Print detailed scores for each agent
        for agent_name, details in agent_details.items():
            print(f'\n{agent_name} Detailed Scores:')
            for dimension, score in details.items():
                if dimension != 'overall_score':
                    print(f'  {dimension}: {score}')

        # Print comments
        if results.get('comments'):
            print(f'\nEvaluation Comments:\n{results["comments"]}')

    