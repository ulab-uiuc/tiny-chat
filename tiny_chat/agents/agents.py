from typing import Any
from tiny_chat.generator import agenerate_action, agenerate_goal
from tiny_chat.messages import AgentAction, Message, Observation
from tiny_chat.profile import BaseAgentProfile
from tiny_chat.agents.llm_client import LLMGenerator, LLMConfig, LLMBackend


class LLMAgent:
    """LLM-based agent that uses OpenAI API - simplified version"""

    def __init__(
        self,
        agent_name: str,
        model: str = 'gpt-4o-mini',
        agent_number: int = 1,
        api_key: str | None = None,
        goal: str | None = None,
        agent_profile: BaseAgentProfile | None = None,
        script_like: bool = False,
        backend: str = 'litellm',
        source: str = 'openai',
        use_same_model: bool = False,
    ):
        self.agent_name = agent_name
        self.model = model
        self.agent_number = agent_number
        self.api_key = api_key
        self._goal = goal or 'Have a natural conversation'
        self.agent_profile = agent_profile
        self.message_history: list[Message] = []
        self.script_like = script_like
        self.backend = backend
        self.source = source
        self.use_same_model = use_same_model
        
        # 初始化LLM生成器
        if self.use_same_model:
            self.llm_generator = None
        else:
            self.llm_generator = self._create_llm_generator()

    def _create_llm_generator(self) -> LLMGenerator:
        """创建LLM生成器"""
        config = LLMConfig(
            backend=LLMBackend.LITELLM if self.backend == 'litellm' else LLMBackend.VLLM,
            model_name=self.model,
            api_key=self.api_key,
            temperature=0.7,
        )
        return LLMGenerator(config)

    @property
    def goal(self) -> str:
        """Get the agent's goal"""
        return self._goal

    @goal.setter
    def goal(self, goal: str) -> None:
        """Set the agent's goal"""
        self._goal = goal

    async def act(self, obs: Observation, llm_generator: LLMGenerator) -> AgentAction:
        self.receive_message('Environment', obs)

        if len(obs.available_actions) == 1 and 'none' in obs.available_actions:
            return AgentAction(action_type='none', argument='')
        else:
            action = await agenerate_action(
                model_name=self.model,
                history=self._build_context(obs),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
                llm_generator=llm_generator,
            )
            return action

    def _build_context(self, observation: Observation) -> str:
        """Build context string from message history and observation"""
        context_parts = []

        if self.agent_profile:
            context_parts.append(
                f'Agent Profile: {self.agent_profile.to_natural_language()}'
            )

        if self.goal:
            context_parts.append(f'Your goal: {self.goal}')

        for msg in self.message_history:
            context_parts.append(f'{msg.to_natural_language()}')

        context_parts.append(observation.to_natural_language())

        return '\n'.join(context_parts)

    def reset(self) -> None:
        """Reset agent state"""
        self.message_history.clear()

    def receive_message(self, source: str, message: Message) -> None:
        """Receive a message and add to history"""
        self.message_history.append(message)


class TwoAgentManager:
    def __init__(self, agent_configs: list[dict[str, Any]], background=None, api_key=None, general_settings=dict[str, Any]):
        self.agent_configs = agent_configs
        self.background = background
        self.api_key = api_key
        self.general_settings = general_settings
        self.shared_llm_generator = None
        self.agents = {}

    async def initialize_agents(self):
        if self.general_settings.get('use_same_model', False):
            self.agents = await self._create_agents_with_same_models()
        else:
            self.agents = await self._create_agents_with_different_models()

    async def _create_agents_with_different_models(self):
        agents = {}
        for config in self.agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']
            agent_number = config.get('agent_number', 1)
            backend = config.get('backend', 'litellm')
            source = config.get('source', 'openai')
            
            if agent_type == 'llm':
                model = config.get('model', 'gpt-4o-mini')
                agent = LLMAgent(
                    agent_name=name,
                    agent_number=agent_number,
                    model=model,
                    api_key=self.api_key,
                    backend=backend,
                    source=source,
                )
            # elif agent_type == 'human':
            #     agent = HumanAgent(
            #         agent_name=name,
            #         agent_number=agent_number,
            #     )
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            if 'goal' in config:
                agent.goal = config['goal']
            elif self.background and agent_type == 'llm':
                agent.goal = await agenerate_goal(
                    model_name='gpt-4o-mini',
                    background=self.background.to_natural_language(),
                    llm_generator=agent.llm_generator,
                )

            agents[name] = agent
            
        return agents

    async def _create_agents_with_same_models(self):
        first_config = self.agent_configs[0]
        shared_model = first_config.get('model', 'gpt-4o-mini')
        shared_source = first_config.get('source', 'open_source')
        
        for i, config in enumerate(self.agent_configs):
            if config.get('model', 'gpt-4o-mini') != shared_model:
                raise ValueError(f'Agent {i+1} model {config.get("model")} does not match shared model {shared_model}')
        
        self.shared_llm_generator = self._create_shared_llm_generator(shared_model, shared_source)

        agents = {}
        for config in self.agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']
            agent_number = config.get('agent_number', 1)

            if agent_type == 'llm':
                agent = LLMAgent(
                    agent_name=name,
                    agent_number=agent_number,
                    model=shared_model,
                    api_key=self.api_key,
                    use_same_model=True,
                )
            # elif agent_type == 'human':
            #     agent = HumanAgent(
            #         agent_name=name,
            #         agent_number=agent_number,
            #     )
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            if 'goal' in config:
                agent.goal = config['goal']
            elif self.background and agent_type == 'llm':
                agent.goal = await agenerate_goal(
                    model_name=shared_model,
                    background=self.background.to_natural_language(),
                    llm_generator=self.shared_llm_generator,
                )

            agents[name] = agent

        return agents

    def _create_shared_llm_generator(self, model: str, source: str) -> LLMGenerator:
        config = LLMConfig(
            backend=LLMBackend.LITELLM if source == 'open_source' else LLMBackend.VLLM,
            model_name=model,
            api_key=self.api_key,
            temperature=0.7,
        )
        return LLMGenerator(config)

    def _reset_agents(self):
        for agent in self.agents.values():
            agent.reset()


class MultiAgentManager:
    
    def __init__(self, agent_configs: list[dict[str, Any]], background=None, api_key=None, general_settings=dict[str, Any]):
        self.agent_configs = agent_configs
        self.background = background
        self.api_key = api_key
        self.general_settings = general_settings
        self.agents = {}

    async def initialize_agents(self):
        """异步初始化所有代理"""
        self.agents = await self._create_agents()

    async def _create_agents(self):
        """创建所有代理，支持模型共享优化"""
        agents = {}
        llm_generators = {}  # 缓存LLM生成器
        
        for config in self.agent_configs:
            agent_type = config.get('type', 'llm')
            name = config['name']
            agent_number = config.get('agent_number', 1)
            
            if agent_type == 'llm':
                model = config.get('model', 'gpt-4o-mini')
                backend = config.get('backend', 'litellm')
                source = config.get('source', 'openai')
                
                # 检查是否可以使用共享的LLM生成器
                model_key = f"{model}_{backend}_{source}"
                if model_key in llm_generators:
                    # 使用共享的LLM生成器
                    shared_generator = llm_generators[model_key]
                    agent = LLMAgent(
                        agent_name=name,
                        agent_number=agent_number,
                        model=model,
                        api_key=self.api_key,
                        backend=backend,
                        source=source,
                    )
                    agent._change_llm_generator(shared_generator)
                else:
                    # 创建新的LLM生成器
                    agent = LLMAgent(
                        agent_name=name,
                        agent_number=agent_number,
                        model=model,
                        api_key=self.api_key,
                        backend=backend,
                        source=source,
                    )
                    llm_generators[model_key] = agent.llm_generator
                    
            elif agent_type == 'human':
                agent = HumanAgent(
                    agent_name=name,
                    agent_number=agent_number,
                )
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')

            if 'goal' in config:
                agent.goal = config['goal']
            elif self.background and agent_type == 'llm':
                agent.goal = await agenerate_goal(
                    model_name=model if agent_type == 'llm' else 'gpt-4o-mini',
                    background=self.background.to_natural_language(),
                    llm_generator=agent.llm_generator,
                )

            agents[name] = agent
            
        return agents

    def reset_agents(self):
        """重置所有代理状态"""
        for agent in self.agents.values():
            agent.reset()

    def get_agent_names(self) -> list[str]:
        """获取所有代理名称"""
        return list(self.agents.keys())

    def get_agent(self, name: str):
        """获取指定名称的代理"""
        return self.agents.get(name)


class HumanAgent:
    pass