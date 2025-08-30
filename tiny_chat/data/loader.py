import json
import random
from typing import Any

from datasets import load_dataset

from tiny_chat.profiles.relationship_profile import (
    BaseAgentProfile,
    BaseEnvironmentProfile,
    BaseRelationshipProfile,
    RelationshipType,
)


class DataLoader:
    """a class to load data from hugging face"""

    def __init__(self, use_official: bool = True):
        if use_official:
            self.hf_repo = 'skyyyyks/tiny-chat'
            self.agent_profiles_dataset = 'agent_profiles.jsonl'
            self.env_profiles_dataset = 'environment_profiles.jsonl'
            self.relationship_profiles_dataset = 'relationship_profiles.jsonl'
        self.agent_profiles: Any = None
        self.env_profiles: Any = None
        self.relationship_profiles: Any = None

    def load_agent_profiles(
        self, use_local: bool = False, local_path: str | None = None
    ) -> None:
        if not use_local:
            # Load the dataset from Hugging Face
            self.agent_profiles = load_dataset(
                self.hf_repo, data_files=self.agent_profiles_dataset
            )
        else:
            # Load the dataset from local file
            if local_path is None:
                raise ValueError('local_path must be provided to load local data')
            try:
                with open(local_path) as f:
                    self.agent_profiles = []
                    for line in f:
                        self.agent_profiles.append(json.loads(line.strip()))
            except FileNotFoundError:
                print(f'File not found: {local_path}')
            except json.JSONDecodeError as e:
                print(f'Error decoding JSON: {e}')

    def get_all_agent_profiles(
        self, use_local: bool = False, local_path: str | None = None
    ) -> list[BaseAgentProfile]:
        if self.agent_profiles is None or use_local:
            self.load_agent_profiles(use_local, local_path)

        profiles = []
        for record in self.agent_profiles:
            # Create a BaseAgentProfile instance from the record and add it to the list
            agent_profile = BaseAgentProfile(
                pk=record.get('pk', ''),
                first_name=record.get('first_name', ''),
                last_name=record.get('last_name', ''),
                age=record.get('age', 0),
                occupation=record.get('occupation', ''),
                gender=record.get('gender', ''),
                gender_pronoun=record.get('gender_pronoun', ''),
                public_info=record.get('public_info', ''),
                big_five=record.get('big_five', ''),
                moral_values=record.get('moral_values', []),
                schwartz_personal_values=record.get('schwartz_personal_values', []),
                personality_and_values=record.get('personality_and_values', ''),
                decision_making_style=record.get('decision_making_style', ''),
                secret=record.get('secret', ''),
                model_id=record.get('model_id', ''),
                mbti=record.get('mbti', ''),
                speaking_id=record.get('speaking_id', 0),
            )

            profiles.append(agent_profile)

        return profiles

    def load_env_profiles(
        self, use_local: bool = False, local_path: str | None = None
    ) -> None:
        if not use_local:
            # Load the dataset from Hugging Face
            self.env_profiles = load_dataset(
                self.hf_repo, data_files=self.env_profiles_dataset
            )
        else:
            # Load the dataset from local file
            if local_path is None:
                raise ValueError('local_path must be provided to load local data')
            try:
                with open(local_path) as f:
                    self.env_profiles = []
                    for line in f:
                        self.env_profiles.append(json.loads(line.strip()))
            except FileNotFoundError:
                print(f'File not found: {local_path}')
            except json.JSONDecodeError as e:
                print(f'Error decoding JSON: {e}')

    def get_all_env_profiles(
        self, use_local: bool = False, local_path: str | None = None
    ) -> list[BaseEnvironmentProfile]:
        if self.env_profiles is None or use_local:
            self.load_env_profiles(use_local, local_path)

        profiles = []
        for record in self.env_profiles:
            # Create a BaseEnvironmentProfile instance from the record and add it to the list
            env_profile = BaseEnvironmentProfile(
                pk=record.get('pk', ''),
                codename=record.get('codename', ''),
                source=record.get('source', ''),
                scenario=record.get('scenario', ''),
                agent_goals=record.get('agent_goals', []),
                relationship=record.get('relationship', RelationshipType.stranger),
                age_constraint=record.get('age_constraint', ''),
                occupation_constraint=record.get('occupation_constraint', ''),
                agent_constraint=record.get('agent_constraint', []),
            )

            profiles.append(env_profile)

        return profiles

    # NOTE: currently only support BaseRelationshipProfile as the same of the jsonl file on Hugging Face
    def load_relationship_profiles(
        self, use_local: bool = False, local_path: str | None = None
    ) -> None:
        if not use_local:
            # Load the dataset from Hugging Face
            self.relationship_profiles = load_dataset(
                self.hf_repo, data_files=self.relationship_profiles_dataset
            )
        else:
            # Load the dataset from local file
            if local_path is None:
                raise ValueError('local_path must be provided to load local data')
            try:
                with open(local_path) as f:
                    self.relationship_profiles = []
                    for line in f:
                        self.relationship_profiles.append(json.loads(line.strip()))
            except FileNotFoundError:
                print(f'File not found: {local_path}')
            except json.JSONDecodeError as e:
                print(f'Error decoding JSON: {e}')

    def get_all_relationship_profiles(
        self, use_local: bool = False, local_path: str | None = None
    ) -> list[BaseRelationshipProfile]:
        if self.relationship_profiles is None or use_local:
            self.load_relationship_profiles(use_local, local_path)

        profiles = []
        for record in self.relationship_profiles:
            # get all ids
            agent_ids: set[str] = set()
            for key, value in record.items():
                if (
                    key.startswith('agent_')
                    and key.endswith('_id')
                    and isinstance(value, str)
                ):
                    agent_ids.add(value)

            # Create a BaseRelationshipProfile instance from the record and add it to the list
            rel_profile = BaseRelationshipProfile(
                pk=record.get('pk', ''),
                agent_ids=agent_ids,
                default_relationship=record.get(
                    'relationship', RelationshipType.stranger
                ),
                scenario_context=record.get('background_story', ''),
            )

            profiles.append(rel_profile)

        return profiles
