import logging
from contextlib import asynccontextmanager
from typing import Any

from tiny_chat.agents import LLMAgent
from tiny_chat.config import ServerConfig, get_config
from tiny_chat.envs import TinyChatEnvironment
from tiny_chat.evaluator import EvaluatorManager
from tiny_chat.messages import TinyChatBackground
from tiny_chat.providers import BaseModelProvider, ModelProviderFactory
from tiny_chat.utils import EpisodeLog, save_conversation_to_json, setup_logging

logger = logging.getLogger(__name__)


class TinyChatServer:
    """Main TinyChat server with dependency injection"""

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or get_config()
        self.model_providers: dict[str, BaseModelProvider] = {}
        self.evaluator_manager = EvaluatorManager()
        self._setup_logging()

    async def initialize(self) -> None:
        """Initialize the server"""
        logger.info('Initializing TinyChat Server...')

        await self._load_model_providers()
        await self._load_evaluators()

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

    async def _load_evaluators(self) -> None:
        for evaluator_config in self.config.evaluators:
            if not evaluator_config.enabled:
                continue

            try:
                from ..evaluator import EvaluatorConfig

                eval_config = EvaluatorConfig(
                    type=evaluator_config.type,
                    config=evaluator_config.config,
                    model=evaluator_config.model,
                    enabled=evaluator_config.enabled,
                )

                evaluator = self.evaluator_manager.create_evaluator(
                    eval_config, self.model_providers
                )
                if evaluator:
                    logger.info(f'Loaded evaluator: {evaluator_config.type}')
            except Exception as e:
                logger.error(f'Failed to load evaluator {evaluator_config.type}: {e}')

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        setup_logging()

    def _neighbor_map_ids_to_names(
        self,
        agent_configs: list[dict[str, Any]],
        id_map: dict[int, list[int]] | None,
    ) -> dict[str, list[str]] | None:
        if not id_map:
            return None

        id_to_name: dict[int, str] = {}
        for cfg in agent_configs:
            sid = cfg.get('speaking_id')
            if isinstance(sid, int):
                id_to_name[sid] = cfg['name']

        name_map: dict[str, list[str]] = {}
        for sid, nbr_ids in id_map.items():
            if sid not in id_to_name:
                continue
            src_name = id_to_name[sid]
            nbr_names = [id_to_name[nid] for nid in nbr_ids if nid in id_to_name]
            if nbr_names:
                name_map[src_name] = nbr_names
        return name_map or None

    async def run_conversation(
        self,
        agent_configs: list[dict[str, Any]],
        background: TinyChatBackground | None = None,
        max_turns: int | None = None,
        enable_evaluation: bool = True,
        action_order: str | None = None,
        speaking_order: list[int] | None = None,
        scenario: str | None = None,
        default_model: str | None = None,
        return_log: bool = False,
        obs_control: dict[str, Any] | None = None,
    ) -> EpisodeLog | None:
        max_turns = max_turns or self.config.max_turns
        action_order = action_order or self.config.action_order
        speaking_order = speaking_order or getattr(self.config, 'speaking_order', None)
        default_model_name = default_model or self.config.default_model

        self._validate_models(agent_configs, default_model_name)

        logger.info(f'Starting conversation with {len(agent_configs)} agents')

        evaluators, terminal_evaluators = self._create_evaluators(
            max_turns, enable_evaluation
        )
        raw_id_neighbor_map = (obs_control or {}).get('neighbor_map')

        neighbor_map_names = self._neighbor_map_ids_to_names(
            agent_configs=agent_configs,
            id_map=raw_id_neighbor_map,
        )
        env = TinyChatEnvironment(
            evaluators=evaluators,
            terminal_evaluators=terminal_evaluators,
            action_order=action_order,
            speaking_order=speaking_order,
            max_turns=max_turns,
            available_action_types=set(self.config.available_action_types),
            obs_mode=(obs_control or {}).get('mode', 'all'),
            neighbor_map=neighbor_map_names,
            local_k=(obs_control or {}).get('local_k', 5),
        )

        agents = await self._create_agents(
            agent_configs, background, default_model_name
        )

        env.reset(
            agents=agents,
            options={
                'scenario': scenario or (background.scenario if background else None)
            },
        )

        logger.info(
            f"Starting {len(agents)}-agent conversation in {'sync' if self.config.sync_mode else 'async'} mode"
        )
        if self.config.sync_mode:
            self._run_conversation_loop_sync(env, agents)
        else:
            await self._run_conversation_loop(env, agents)
        logger.info(f'Conversation ended after {env.get_turn_number()} turns')

        evaluation_results = {}
        if enable_evaluation and terminal_evaluators:
            if self.config.sync_mode:
                evaluation_results = env.evaluate_episode_sync()
            else:
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
        for config in agent_configs:
            name = config['name']
            agent_provider_key = config.get('model_provider')

            if agent_provider_key and agent_provider_key not in self.model_providers:
                raise ValueError(
                    f"Model provider '{agent_provider_key}' for agent '{name}' not available. "
                    f'Available providers: {list(self.model_providers.keys())}'
                )

    def _create_evaluators(
        self, max_turns: int, enable_evaluation: bool
    ) -> tuple[list, list]:
        evaluators = []
        terminal_evaluators = []

        for evaluator in self.evaluator_manager.get_evaluators():
            evaluator_type = evaluator.evaluator_type

            if evaluator_type == 'rule_based':
                evaluators.append(evaluator)
            elif evaluator_type == 'llm' and enable_evaluation:
                if hasattr(evaluator, 'get_terminal_evaluator'):
                    term = evaluator.get_terminal_evaluator()
                    if term:
                        terminal_evaluators.append(term)
            else:
                evaluators.append(evaluator)

        if not any(
            evaluator.evaluator_type == 'rule_based' for evaluator in evaluators
        ):
            logger.warning(
                'No rule-based evaluator configured, conversation may not terminate properly'
            )

        return evaluators, terminal_evaluators

    async def _create_agents(
        self,
        agent_configs: list[dict[str, Any]],
        background: TinyChatBackground | None,
        default_model_name: str,
    ) -> dict[str, Any]:
        """
        Create agents with simplified model_provider configuration

        Agent config format:
        {
            "name": "Alice",
            "type": "llm",
            "model_provider": "provider_key",  # optional: provider from config, uses default if not specified
            "goal": "Agent's goal",            # optional: generated by background
            "script_like": False,              # optional: generation mode
        }
        """
        agents = {}

        for config in agent_configs:
            name = config['name']
            agent_type = config.get('type', 'llm')
            agent_provider_key = config.get('model_provider')

            if agent_type != 'llm':
                raise ValueError(f'Unsupported agent type: {agent_type}')

            if agent_provider_key:
                if agent_provider_key not in self.model_providers:
                    raise ValueError(
                        f"Model provider '{agent_provider_key}' for agent '{name}' not available. "
                        f'Available providers: {list(self.model_providers.keys())}'
                    )
                provider = self.model_providers[agent_provider_key]
            else:
                if default_model_name not in self.model_providers:
                    provider = None
                else:
                    provider = self.model_providers[default_model_name]

            logger.info(
                f"Creating {name} with provider '{type(provider).__name__ if provider else 'default'}'"
            )

            agent = LLMAgent(
                agent_name=name,
                model_provider=provider,
                script_like=config.get('script_like', False),
            )

            if 'speaking_id' in config:
                if not hasattr(agent, 'profile') or agent.profile is None:
                    from tiny_chat.profiles import BaseAgentProfile

                    agent.profile = BaseAgentProfile(
                        first_name=name,
                        last_name='',
                        speaking_id=config['speaking_id'],
                        occupation='',
                        public_info='',
                    )
                else:
                    agent.profile.speaking_id = config['speaking_id']

            if 'goal' in config:
                agent.goal = config['goal']
            elif background:
                agent.goal = await agent._model_provider.agenerate_goal(
                    background=background.to_natural_language()
                )

            agents[name] = agent

        return agents

    def _run_conversation_loop_sync(
        self, env: TinyChatEnvironment, agents: dict[str, Any]
    ) -> None:
        """Run the conversation loop synchronously"""
        while not env.is_terminated():
            turn_num = env.get_turn_number()
            logger.info(f'--- Turn {turn_num} ---')

            actions = {}

            if env.action_order == 'agent_id_based' and env.speaking_order:
                for agent_name in env.agent_names:
                    if agent_name in agents:
                        agent = agents[agent_name]
                        observation = env.get_observation(agent_name)
                        action = agent.act_sync(observation)
                        actions[agent_name] = action
                        logger.info(f'{agent_name}: {action.to_natural_language()}')
            else:
                for name, agent in agents.items():
                    observation = env.get_observation(name)
                    action = agent.act_sync(observation)
                    actions[name] = action
                    logger.info(f'{name}: {action.to_natural_language()}')

            env.step(actions)

    async def _run_conversation_loop(
        self, env: TinyChatEnvironment, agents: dict[str, Any]
    ) -> None:
        """Run the conversation loop asynchronously"""
        while not env.is_terminated():
            turn_num = env.get_turn_number()
            logger.info(f'--- Turn {turn_num} ---')

            actions = {}

            if env.action_order == 'agent_id_based' and env.speaking_order:
                for agent_name in env.agent_names:
                    if agent_name in agents:
                        agent = agents[agent_name]
                        observation = env.get_observation(agent_name)
                        action = await agent.act(observation)
                        actions[agent_name] = action
                        logger.info(f'{agent_name}: {action.to_natural_language()}')
            else:
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

        logger.info(f'Conversation terminated: {results.get("terminated", False)}')

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
