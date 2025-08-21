import logging
from contextlib import asynccontextmanager
from typing import Any

from ..agents import LLMAgent
from ..envs import TinyChatEnvironment
from ..evaluator import RuleBasedTerminatedEvaluator
from ..messages import TinyChatBackground
from ..utils import EpisodeLog
from ..utils.json_saver import save_conversation_to_json
from ..utils.logger import setup_logging
from .config import ServerConfig, get_config
from .plugins import PluginManager
from .providers import BaseModelProvider, ModelProviderFactory

logger = logging.getLogger(__name__)


class TinyChatServer:
    """Main TinyChat server with dependency injection"""

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or get_config()
        self.model_providers: dict[str, BaseModelProvider] = {}
        self.plugin_manager = PluginManager()
        self._setup_logging()

    async def initialize(self) -> None:
        """Initialize the server"""
        logger.info('Initializing TinyChat Server...')

        await self._load_model_providers()
        await self._load_plugins()

        logger.info(
            f'Server initialized with {len(self.model_providers)} model providers'
        )

    async def _load_model_providers(self) -> None:
        """Load and initialize model providers"""
        for name, config in self.config.models.items():
            try:
                provider = ModelProviderFactory.create_provider(config)

                self.model_providers[name] = provider
            except Exception as e:
                logger.error(f'Failed to load provider {name}: {e}')

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
        setup_logging()

    async def run_conversation(
        self,
        agent_configs: list[dict[str, Any]],
        background: TinyChatBackground | None = None,
        max_turns: int | None = None,
        enable_evaluation: bool = True,
        action_order: str | None = None,
        scenario: str | None = None,
        default_model: str | None = None,
        return_log: bool = False,
    ) -> EpisodeLog | None:
        max_turns = max_turns or self.config.max_turns
        action_order = action_order or self.config.action_order
        default_model_name = default_model or self.config.default_model

        self._validate_models(agent_configs, default_model_name)

        logger.info(f'Starting conversation with {len(agent_configs)} agents')

        evaluators, terminal_evaluators = self._create_evaluators(
            max_turns, enable_evaluation
        )

        env = TinyChatEnvironment(
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
            action_order=action_order,
            max_turns=max_turns,
            available_action_types=set(self.config.available_action_types),
        )

        agents = await self._create_agents(
            agent_configs, background, default_model_name
        )

        env.reset(
            agents=agents,
            options={
                'scenario': scenario
                or (background.to_natural_language() if background else None)
            },
        )

        logger.info(f'Starting {len(agents)}-agent conversation')
        await self._run_conversation_loop(env, agents)
        logger.info(f'Conversation ended after {env.get_turn_number()} turns')

        evaluation_results = {}
        if enable_evaluation and terminal_evaluators:
            evaluation_results = await env.evaluate_episode()
            self._log_evaluation_results(evaluation_results)

        self._save_conversation_log(
            env,
            agent_configs,
            evaluation_results,
            scenario,
            background,
            max_turns,
            action_order,
        )

        return self._create_episode_log(agent_configs, evaluation_results)

    def _validate_models(
        self, agent_configs: list[dict[str, Any]], default_model_name: str
    ) -> None:
        if default_model_name not in self.model_providers:
            raise ValueError(f"Default model '{default_model_name}' not available")

        # 验证每个 Agent 的模型
        for config in agent_configs:
            agent_model = config.get('model_name', default_model_name)
            if agent_model not in self.model_providers:
                raise ValueError(
                    f"Model '{agent_model}' for agent '{config['name']}' not available"
                )

    def _create_evaluators(
        self, max_turns: int, enable_evaluation: bool
    ) -> tuple[list, list]:
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=max_turns)]
        terminal_evaluators = []

        if enable_evaluation:
            for plugin in self.plugin_manager.get_evaluators():
                if hasattr(plugin, 'get_terminal_evaluator'):
                    terminal_eval = plugin.get_terminal_evaluator()
                    if terminal_eval:
                        terminal_evaluators.append(terminal_eval)
                else:
                    evaluators.append(plugin)

        return evaluators, terminal_evaluators

    async def _create_agents(
        self,
        agent_configs: list[dict[str, Any]],
        background: TinyChatBackground | None,
        default_model_name: str,
    ) -> dict[str, Any]:
        """
        🎯 Create agents with individual ModelProvider bindings

        Agent config format：
        {
            "name": "Alice",
            "type": "llm",
            "model_name": "gpt-4o",  # optional, default to default_model
            "goal": "Agent's goal",   # optional, generated by background
        }
        """
        agents = {}

        for config in agent_configs:
            name = config['name']
            agent_type = config.get('type', 'llm')
            agent_model_name = config.get('model_name', default_model_name)

            if agent_type != 'llm':
                raise ValueError(f'Unsupported agent type: {agent_type}')

            provider = self.model_providers[agent_model_name]
            logger.info(
                f'Creating {name} with {agent_model_name} ({type(provider).__name__})'
            )

            agent = LLMAgent(
                agent_name=name,
                model_name=agent_model_name,
                model_provider=provider,
                script_like=config.get('script_like', False),
            )

            if 'goal' in config:
                agent.goal = config['goal']
            elif background:
                agent.goal = await provider.agenerate_goal(
                    background=background.to_natural_language()
                )

            agents[name] = agent

        return agents

    async def _run_conversation_loop(
        self, env: TinyChatEnvironment, agents: dict[str, Any]
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

    def _log_evaluation_results(self, results: dict[str, Any]) -> None:
        """Log evaluation results"""
        if 'message' in results:
            logger.info(f'Evaluation: {results["message"]}')
            return

        logger.info(f"Conversation terminated: {results.get('terminated', False)}")

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

    def _save_conversation_log(
        self,
        env: TinyChatEnvironment,
        agent_configs: list[dict[str, Any]],
        evaluation_results: dict[str, Any],
        scenario: str | None,
        background: TinyChatBackground | None,
        max_turns: int,
        action_order: str,
    ) -> None:
        try:
            conversation_history = []
            if hasattr(env, 'inbox') and env.inbox:
                for source, message in env.inbox:
                    if source != 'Environment':
                        conversation_history.append(
                            {
                                'agent': source,
                                'content': message.to_natural_language(),
                                'turn': len(conversation_history) + 1,
                            }
                        )

            agent_profile = {
                config['name']: {
                    'name': config['name'],
                    'type': config.get('type', 'llm'),
                    'model': config.get('model_name', 'default'),
                    'goal': config.get('goal', ''),
                }
                for config in agent_configs
            }

            environment_profile = {
                'scenario': scenario
                or (background.to_natural_language() if background else ''),
                'max_turns': max_turns,
                'action_order': action_order,
                'total_turns': env.get_turn_number(),
                'agents': [config['name'] for config in agent_configs],
            }

            save_conversation_to_json(
                agent_profile=agent_profile,
                environment_profile=environment_profile,
                conversation_history=conversation_history,
                evaluation=evaluation_results,
                output_dir='conversation_logs',
            )
            logger.info('Conversation saved to conversation_logs/')

        except Exception as e:
            logger.warning(f'Failed to save conversation: {e}')

    def _create_episode_log(
        self, agent_configs: list[dict[str, Any]], evaluation_results: dict[str, Any]
    ) -> EpisodeLog:
        """Create episode log from results"""
        rewards = []
        for config in agent_configs:
            agent_name = config['name']
            score = evaluation_results.get(f'{agent_name}_score', 0.0)
            details = evaluation_results.get(f'{agent_name}_details', {})
            rewards.append((score, details))

        return EpisodeLog(
            environment='TinyChat',
            agents=[c['name'] for c in agent_configs],
            messages=[],
            rewards=rewards,
            reasoning=evaluation_results.get('comments', ''),
            models=[
                c.get('model_name', self.config.default_model) for c in agent_configs
            ],
        )

    async def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        if model_name:
            if model_name not in self.model_providers:
                raise ValueError(f"Model '{model_name}' not found")

            provider = self.model_providers[model_name]
            return {
                'name': provider.name,
                'type': provider.type,
                'config': provider.config.dict(),
                'healthy': await provider.check_health(),
            }
        else:
            return {
                name: {
                    'name': provider.name,
                    'type': provider.type,
                    'healthy': await provider.check_health(),
                }
                for name, provider in self.model_providers.items()
            }

    async def shutdown(self) -> None:
        logger.info('Shutting down TinyChat Server...')
        self.model_providers.clear()
        logger.info('Server shutdown complete')


@asynccontextmanager
async def create_server(config: ServerConfig | None = None):
    server = TinyChatServer(config)
    try:
        await server.initialize()
        yield server
    finally:
        await server.shutdown()
