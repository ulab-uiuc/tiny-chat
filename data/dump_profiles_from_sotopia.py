import json

from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)


def dump_agent_profiles_to_jsonl(output_file: str = "agent_profiles.jsonl") -> None:
    """Dump agent profiles from Sotopia database to JSONL file"""
    agents = AgentProfile.all()
    if not agents:
        raise ValueError("No agent profiles found in Redis!")

    with open(output_file, "w", encoding="utf-8") as f:
        for agent in agents:
            data = agent.dict() if hasattr(agent, "dict") else agent.__dict__
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Dumped {len(agents)} agent profiles into {output_file}")


def dump_environment_profiles_to_jsonl(
    output_file: str = "environment_profiles.jsonl",
) -> None:
    """Dump environment profiles from Sotopia database to JSONL file"""
    envs = EnvironmentProfile.all()
    if not envs:
        raise ValueError("No environment profiles found in Redis!")

    with open(output_file, "w", encoding="utf-8") as f:
        for env in envs:
            data = env.dict() if hasattr(env, "dict") else env.__dict__
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Dumped {len(envs)} environment profiles into {output_file}")


def dump_relationship_profiles_to_jsonl(
    output_file: str = "relationship_profiles.jsonl",
) -> None:
    """Dump relationship profiles from Sotopia database to JSONL file"""
    relationships = RelationshipProfile.all()
    if not relationships:
        raise ValueError("No relationship profiles found in Redis!")

    with open(output_file, "w", encoding="utf-8") as f:
        for rel in relationships:
            data = rel.dict() if hasattr(rel, "dict") else rel.__dict__
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Dumped {len(relationships)} relationship profiles into {output_file}")


def dump_all_profiles(
    agent_file: str = "agent_profiles.jsonl",
    environment_file: str = "environment_profiles.jsonl",
    relationship_file: str = "relationship_profiles.jsonl",
) -> None:
    """Dump all profile types from Sotopia database to JSONL files"""
    print("Dumping all profiles from Sotopia database...")

    try:
        dump_agent_profiles_to_jsonl(agent_file)
    except Exception as e:
        print(f"Failed to dump agent profiles: {e}")

    try:
        dump_environment_profiles_to_jsonl(environment_file)
    except Exception as e:
        print(f"Failed to dump environment profiles: {e}")

    try:
        dump_relationship_profiles_to_jsonl(relationship_file)
    except Exception as e:
        print(f"Failed to dump relationship profiles: {e}")

    print("Profile dumping completed!")
