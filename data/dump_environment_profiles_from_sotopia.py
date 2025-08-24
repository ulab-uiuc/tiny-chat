import json

from sotopia.database.persistent_profile import EnvironmentProfile


def dump_environment_profiles_to_jsonl(output_file: str = 'environment_profiles.jsonl'):
    envs = EnvironmentProfile.all()
    if not envs:
        raise ValueError('No environment profiles found in Redis!')

    with open(output_file, 'w', encoding='utf-8') as f:
        for env in envs:
            data = env.dict() if hasattr(env, 'dict') else env.__dict__

            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f'Dumped {len(envs)} environment profiles into {output_file}')


if __name__ == '__main__':
    dump_environment_profiles_to_jsonl('environment_profiles.jsonl')
