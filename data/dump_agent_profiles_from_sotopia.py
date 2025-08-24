import json

from sotopia.database.persistent_profile import AgentProfile


def dump_agent_profiles_to_jsonl(output_file: str = 'agent_profiles.jsonl'):
    agents = AgentProfile.all()
    if not agents:
        raise ValueError('No agent profiles found in Redis!')

    with open(output_file, 'w', encoding='utf-8') as f:
        for agent in agents:
            data = agent.dict() if hasattr(agent, 'dict') else agent.__dict__

            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f'Dumped {len(agents)} agent profiles into {output_file}')


if __name__ == '__main__':
    dump_agent_profiles_to_jsonl('agent_profiles.jsonl')
