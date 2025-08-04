"""
Simple Chat Server - Run multi-agent conversations
"""

import asyncio
from typing import Dict, List, Optional
from .environment import ChatEnvironment
from .agents import LLMAgent, HumanAgent
from .messages import ChatBackground
from .generator import MessageGenerator
from .evaluators import (
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    EvaluationForTwoAgents,
)


class ChatServer:
    """Simple server to run multi-agent conversations"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.generator = MessageGenerator(api_key=api_key)

    async def run_conversation(
        self,
        agent_configs: List[Dict],
        background: Optional[ChatBackground] = None,
        max_turns: int = 20,
        enable_evaluation: bool = True,
    ):
        """Run a conversation with specified agents"""

        # Create evaluators
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=max_turns)]
        terminal_evaluators = []

        if enable_evaluation:
            terminal_evaluators.append(EpisodeLLMEvaluator(model_name="gpt-4o-mini"))

        # Create environment
        env = ChatEnvironment(
            background=background,
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
        )
        env.max_turns = max_turns

        # Create and add agents
        agents = {}
        for config in agent_configs:
            agent_type = config.get("type", "llm")
            name = config["name"]
            agent_number = config.get("agent_number", 1)

            if agent_type == "llm":
                model = config.get("model", "gpt-4o-mini")
                agent = LLMAgent(
                    name=name,
                    agent_number=agent_number,
                    model=model,
                    api_key=self.api_key,
                )
            elif agent_type == "human":
                agent = HumanAgent(name=name, agent_number=agent_number)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Set goal if provided
            if "goal" in config:
                agent.goal = config["goal"]
            elif background and agent_type == "llm":
                # Generate goal from background
                agent.goal = await self.generator.generate_goal(
                    background.to_natural_language()
                )

            agents[name] = agent
            env.add_agent(agent)

        # Reset environment
        env.reset()

        print("=== Conversation Starting ===")
        if background:
            print(background.to_natural_language())
        print("=" * 50)

        # Main conversation loop
        while not env.is_terminated():
            print(f"\n--- Turn {env.get_turn_number()} ---")

            # Get actions from all agents
            actions = {}
            for name, agent in agents.items():
                observation = env.get_observation(name)
                action = await agent.act(observation)
                actions[name] = action

                # Print the action
                print(f"{name}: {action.to_natural_language()}")

            # Execute step
            await env.step(actions)

        print("\n=== Conversation Ended ===")
        print(f"Total turns: {env.get_turn_number()}")

        # Run episode evaluation if enabled
        if enable_evaluation and terminal_evaluators:
            print("\n=== Running Episode Evaluation ===")
            evaluation_results = await env.evaluate_episode()
            self._print_evaluation_results(evaluation_results)

        print("\n=== Full Conversation Summary ===")
        print(env.get_conversation_summary())

    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results in a formatted way"""
        if "message" in results:
            print(f"Evaluation: {results['message']}")
            return

        print(f"Conversation Terminated: {results.get('terminated', False)}")

        # Print scores for all agents
        agent_scores = {}
        agent_details = {}

        for key, value in results.items():
            if key.endswith("_score"):
                agent_name = key[:-6]  # Remove '_score' suffix
                agent_scores[agent_name] = value
            elif key.endswith("_details"):
                agent_name = key[:-8]  # Remove '_details' suffix
                agent_details[agent_name] = value

        # Print overall scores
        for agent_name, score in agent_scores.items():
            print(f"{agent_name} Overall Score: {score:.2f}")

        # Print detailed scores for each agent
        for agent_name, details in agent_details.items():
            print(f"\n{agent_name} Detailed Scores:")
            for dimension, score in details.items():
                if dimension != "overall_score":
                    print(f"  {dimension}: {score}")

        # Print comments
        if results.get("comments"):
            print(f"\nEvaluation Comments:\n{results['comments']}")

    async def run_simple_demo(self):
        """Run a simple demo conversation"""

        # Create a simple background
        background = ChatBackground.create_two_agent_background(
            scenario="Two friends meeting at a coffee shop",
            agent1_name="Alice",
            agent2_name="Bob",
            agent1_goal="Catch up with Bob and share recent news",
            agent2_goal="Listen to Alice and share own updates",
        )

        # Agent configurations
        agent_configs = [
            {
                "type": "llm",
                "name": "Alice",
                "agent_number": 1,
                "model": "gpt-4o-mini",
                "goal": "Catch up with Bob and share recent news",
            },
            {"type": "human", "name": "Bob", "agent_number": 2},
        ]

        await self.run_conversation(
            agent_configs=agent_configs,
            background=background,
            max_turns=10,
            enable_evaluation=True,
        )


async def main():
    """Main function to run the chat server"""
    import os

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Warning: OPENAI_API_KEY not set. LLM agents will not work properly.")

    server = ChatServer(api_key=api_key)

    # Run demo
    await server.run_simple_demo()


if __name__ == "__main__":
    asyncio.run(main())
