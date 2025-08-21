import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from ..agents import LLMAgent
from ..envs import TinyChatEnvironment
from ..evaluator import (
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    TinyChatDimensions,
)
from ..messages import TinyChatBackground
from ..utils import EpisodeLog
from .config import ServerConfig, get_config
from .plugins import PluginManager
from .providers import BaseModelProvider, ModelProviderFactory

logger = logging.getLogger(__name__)


class TinyChatServer:
    """Main TinyChat server with dependency injection"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or get_config()
        self.model_providers: Dict[str, BaseModelProvider] = {}
        self.plugin_manager = PluginManager()
        self._setup_logging()

    async def initialize(self) -> None:
        """Initialize the server"""
        logger.info('Initializing TinyChat Server...')

        # Load model providers
        await self._load_model_providers()

        # Load plugins
        await self._load_plugins()

        logger.info(
            f'Server initialized with {len(self.model_providers)} model providers'
        )

    async def _load_model_providers(self) -> None:
        """Load and initialize model providers"""
        for name, config in self.config.models.items():
            try:
                provider = ModelProviderFactory.create_provider(config)

                # Check provider health
                if await provider.check_health():
                    self.model_providers[name] = provider
                    logger.info(f'Loaded model provider: {name} ({config.type})')
                else:
                    # Still load the provider but with warning
                    self.model_providers[name] = provider
                    logger.warning(
                        f'Model provider {name} failed health check, but loaded anyway'
                    )

            except Exception as e:
                logger.error(f'Failed to load model provider {name}: {e}')

    async def _load_plugins(self) -> None:
        """Load evaluator plugins"""
        for evaluator_config in self.config.evaluators:
            if not evaluator_config.enabled:
                continue

            try:
                plugin = self.plugin_manager.create_evaluator(
                    evaluator_config, self.model_providers
                )
                if plugin:
                    logger.info(f'Loaded evaluator plugin: {evaluator_config.type}')
            except Exception as e:
                logger.error(f'Failed to load evaluator {evaluator_config.type}: {e}')

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_config = self.config.logging

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.level), format=log_config.format
        )

        # Setup file logging if specified
        if log_config.file_path:
            file_handler = logging.FileHandler(log_config.file_path)
            file_handler.setFormatter(logging.Formatter(log_config.format))
            logging.getLogger().addHandler(file_handler)

    async def run_conversation(
        self,
        agent_configs: List[Dict[str, Any]],
        background: Optional[TinyChatBackground] = None,
        max_turns: Optional[int] = None,
        enable_evaluation: bool = True,
        action_order: Optional[str] = None,
        scenario: Optional[str] = None,
        model_name: Optional[str] = None,
        return_log: bool = False,
    ) -> Optional[EpisodeLog]:
        """Run a multi-agent conversation"""

        # Use configuration defaults
        max_turns = max_turns or self.config.max_turns
        action_order = action_order or self.config.action_order
        model_name = model_name or self.config.default_model

        # Validate model exists
        if model_name not in self.model_providers:
            raise ValueError(
                f"Model '{model_name}' not available. Available models: {list(self.model_providers.keys())}"
            )

        logger.info(
            f'Starting conversation with {len(agent_configs)} agents using model {model_name}'
        )

        # Create evaluators
        evaluators = []
        terminal_evaluators = []

        # Add rule-based evaluator
        evaluators.append(RuleBasedTerminatedEvaluator(max_turn_number=max_turns))

        # Add LLM evaluators if enabled
        if enable_evaluation:
            for plugin in self.plugin_manager.get_evaluators():
                if hasattr(plugin, 'get_terminal_evaluator'):
                    terminal_eval = plugin.get_terminal_evaluator()
                    if terminal_eval is not None:
                        terminal_evaluators.append(terminal_eval)
                else:
                    evaluators.append(plugin)

        # Create environment
        env = TinyChatEnvironment(
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
            action_order=action_order,
            max_turns=max_turns,
            available_action_types=set(self.config.available_action_types),
        )

        # Create agents with dependency injection
        agents = await self._create_agents(agent_configs, background, model_name)

        # Setup environment
        reset_options = {}
        if scenario:
            reset_options['scenario'] = scenario
        elif background:
            reset_options['scenario'] = background.to_natural_language()

        env.reset(agents=agents, options=reset_options if reset_options else None)

        # Run conversation
        num_agents = len(agents)
        logger.info(f'=== {num_agents}-Agent Conversation Starting ===')

        if background:
            logger.info(background.to_natural_language())
        elif scenario:
            logger.info(scenario)

        await self._run_conversation_loop(env, agents)

        logger.info(f'=== {len(agents)}-Agent Conversation Ended ===')
        logger.info(f'Total turns: {env.get_turn_number()}')

        # Run evaluation
        evaluation_results = {}
        if enable_evaluation and terminal_evaluators:
            logger.info('=== Running Episode Evaluation ===')
            evaluation_results = await env.evaluate_episode()
            self._log_evaluation_results(evaluation_results)

        # Return log if requested
        if return_log:
            return self._create_episode_log(agent_configs, evaluation_results)

        return None

    async def _create_agents(
        self,
        agent_configs: List[Dict[str, Any]],
        background: Optional[TinyChatBackground],
        model_name: str,
    ) -> Dict[str, Any]:
        """Create agents with dependency injection"""
        agents = {}
        provider = self.model_providers[model_name]

        for config in agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']

            if agent_type == 'llm':
                agent = LLMAgent(
                    agent_name=name,
                    model_name=model_name,
                )

                # Inject model provider (future enhancement)
                # agent.set_model_provider(provider)

            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            # Set goal if provided
            if 'goal' in config:
                agent.goal = config['goal']
            elif background and agent_type == 'llm':
                # Generate goal using the provider
                from ..generator import agenerate_goal

                agent.goal = await agenerate_goal(
                    model_name=model_name,
                    background=background.to_natural_language(),
                )

            agents[name] = agent

        return agents

    async def _run_conversation_loop(
        self, env: TinyChatEnvironment, agents: Dict[str, Any]
    ) -> None:
        """Run the conversation loop"""
        while not env.is_terminated():
            turn_num = env.get_turn_number()
            logger.info(f'--- Turn {turn_num} ---')

            actions = {}
            for name, agent in agents.items():
                observation = env.get_observation(name)
                action = await agent.act(observation)
                actions[name] = action

                logger.info(f'{name}: {action.to_natural_language()}')

            await env.astep(actions)

    def _log_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Log evaluation results"""
        if 'message' in results:
            logger.info(f'Evaluation: {results["message"]}')
            return

        logger.info(f'Conversation Terminated: {results.get("terminated", False)}')

        # Log agent scores
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
            logger.info(f'{agent_name} Overall Score: {score:.2f}')

        for agent_name, details in agent_details.items():
            logger.info(f'{agent_name} Detailed Scores:')
            for dimension, score in details.items():
                if dimension != 'overall_score':
                    logger.info(f'  {dimension}: {score}')

        if results.get('comments'):
            logger.info(f'Evaluation Comments: {results["comments"]}')

    def _create_episode_log(
        self, agent_configs: List[Dict[str, Any]], evaluation_results: Dict[str, Any]
    ) -> EpisodeLog:
        """Create episode log from results"""
        rewards = []
        for config in agent_configs:
            agent_name = config['name']
            score_key = f'{agent_name}_score'
            details_key = f'{agent_name}_details'

            if score_key in evaluation_results:
                rewards.append(
                    (
                        evaluation_results[score_key],
                        evaluation_results.get(details_key, {}),
                    )
                )
            else:
                rewards.append((0.0, {}))

        return EpisodeLog(
            environment='TinyChat',
            agents=[c['name'] for c in agent_configs],
            messages=[],  # TODO: Extract from environment
            rewards=rewards,
            reasoning=evaluation_results.get('comments', ''),
            models=[c.get('model', self.config.default_model) for c in agent_configs],
        )

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about available models"""
        if model_name:
            if model_name not in self.model_providers:
                raise ValueError(f"Model '{model_name}' not found")

            provider = self.model_providers[model_name]
            health = await provider.check_health()

            return {
                'name': provider.name,
                'type': provider.type,
                'config': provider.config.dict(),
                'healthy': health,
            }
        else:
            # Return info for all models
            models = {}
            for name, provider in self.model_providers.items():
                health = await provider.check_health()
                models[name] = {
                    'name': provider.name,
                    'type': provider.type,
                    'healthy': health,
                }
            return models

    async def shutdown(self) -> None:
        """Shutdown the server"""
        logger.info('Shutting down TinyChat Server...')

        # TODO: Cleanup resources
        self.model_providers.clear()

        logger.info('Server shutdown complete')


# Server lifecycle management
@asynccontextmanager
async def create_server(config: Optional[ServerConfig] = None):
    """Create and manage server lifecycle"""
    server = TinyChatServer(config)
    try:
        await server.initialize()
        yield server
    finally:
        await server.shutdown()
