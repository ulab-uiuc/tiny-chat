import asyncio
from pathlib import Path

from tiny_chat.data import UniformSampler, DataLoader
from tiny_chat.agents import LLMAgent
from tiny_chat import ConfigManager, TinyChatServer, TinyChatBackground


async def main():
    loader = DataLoader(use_official=False)
    loader.load_agent_profiles(use_local=True, local_path="data/agent_profiles.jsonl")
    loader.load_env_profiles(use_local=True, local_path="data/environment_profiles.jsonl")
    loader.load_relationship_profiles(use_local=True, local_path="data/relationship_profiles.jsonl")

    agent_profiles = loader.get_all_agent_profiles()
    env_profiles = loader.get_all_env_profiles()

    sampler = UniformSampler(
        agent_candidates=agent_profiles, 
        env_candidates=env_profiles, 
        data_loader=loader
    )

    for env, agents in sampler.sample(agent_classes=LLMAgent, n_agent=2, replacement=True, size=1):
        agents_dict = {agent.agent_name: agent for agent in agents}
        env.reset(agents=agents_dict)
        
        try:
            config_path = Path("config/environments/demo.yaml")
            config_manager = ConfigManager(config_path)
            server_config = config_manager.load_config()
            
            agent_configs = []
            for i, (agent_name, agent) in enumerate(agents_dict.items()):
                agent_configs.append({
                    "name": agent_name,
                    "model_provider": "model1",
                    "speaking_id": i,
                })
            
            scenario = env.env_background.scenario if env.env_background else "A conversation"
            background = TinyChatBackground(
                scenario=scenario,
                agent_configs=agent_configs,
            )
            
            server = TinyChatServer(server_config)
            await server.initialize()
            
            episode_log = await server.run_conversation(
                agent_configs=agent_configs,
                background=background,
                max_turns=4,
                enable_evaluation=True,
                return_log=True,
            )
            
            if episode_log:
                total_turns = getattr(episode_log, "episode_length", None)
                if total_turns is None:
                    total_turns = (
                        len(episode_log.rewards)
                        if hasattr(episode_log, "rewards")
                        else "Unknown"
                    )
                print(f"Total turns: {total_turns}")
            
        except Exception as e:
            print(f"\nConversation failed: {e}")
            print("This is expected if API keys are not set")


if __name__ == "__main__":
    asyncio.run(main())
