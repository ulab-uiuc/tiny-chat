from datasets import load_dataset
import json
from tiny_chat.profiles.agent_profile import BaseAgentProfile
from tiny_chat.profiles.enviroment_profile import BaseEnvironmentProfile
from tiny_chat.profiles.relationship_profile import BaseRelationshipProfile, RelationshipType
import random


class dataLoader:
    """a class to load data from hugging face"""

    def __init__(self, 
                 database_name: str = "skyyyyks/tiny-chat",
                 agent_profiles_file: str = "agent_profiles.jsonl",
                 env_profiles_file: str = "environment_profiles.jsonl",
                 relationship_profiles_file: str = "relationship_profiles.jsonl"
                 ):
        self.database = database_name
        self.agent_profiles_dataset = agent_profiles_file
        self.env_profiles_dataset = env_profiles_file
        self.relationship_profiles_dataset = relationship_profiles_file
        self.agent_profiles = None
        self.env_profiles = None
        self.relationship_profiles = None

    def set_huggingface_dataset(self, 
                                database_name: str = None, 
                                agent_profiles_file: str = None, 
                                env_profiles_file: str = None, 
                                relationship_profiles_file: str = None):
        if database_name:
            self.database = database_name
        if agent_profiles_file:
            self.agent_profiles_dataset = agent_profiles_file
        if env_profiles_file:
            self.env_profiles_dataset = env_profiles_file
        if relationship_profiles_file:
            self.relationship_profiles_dataset = relationship_profiles_file

    def load_agent_profiles(self, local_path: str = None):
        if local_path:
            # Load the dataset from local file
            self.agent_profiles = list
            try:
                with open(local_path, 'r') as f:
                    for line in f:
                        self.agent_profiles.append(json.loads(line.strip()))
            except FileNotFoundError:
                print(f"File not found: {local_path}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            # Load the dataset from Hugging Face
            self.agent_profiles = load_dataset(self.database, data_files=self.agent_profiles_dataset)

    def get_all_agent_profiles(self, local_path: str = None) -> list[BaseAgentProfile]:
        if self.agent_profiles is None:
            self.load_agent_profiles(local_path = local_path)

        profiles = []
        for record in self.agent_profiles:
            # Create a BaseAgentProfile instance from the record and add it to the list
            agent_profile = BaseAgentProfile(
                pk=record.get("pk", ""),
                first_name=record.get("first_name", ""),
                last_name=record.get("last_name", ""),
                age=record.get("age", 0),
                occupation=record.get("occupation", ""),
                gender=record.get("gender", ""),
                gender_pronoun=record.get("gender_pronoun", ""),
                public_info=record.get("public_info", ""),
                big_five=record.get("big_five", ""),
                moral_values=record.get("moral_values", []),
                schwartz_personal_values=record.get("schwartz_personal_values", []),
                personality_and_values=record.get("personality_and_values", ""),
                decision_making_style=record.get("decision_making_style", ""),
                secret=record.get("secret", ""),
                model_id=record.get("model_id", ""),
                mbti=record.get("mbti", "")
            )
            
            profiles.append(agent_profile)

        return profiles
    
    def load_env_profiles(self, local_path: str = None):
        if local_path:
            # Load the dataset from local file
            self.env_profiles = list
            try:
                with open(local_path, 'r') as f:
                    for line in f:
                        self.env_profiles.append(json.loads(line.strip()))
            except FileNotFoundError:
                print(f"File not found: {local_path}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            # Load the dataset from Hugging Face
            self.env_profiles = load_dataset(self.database, data_files=self.env_profiles_dataset)

    def get_all_env_profiles(self, local_path: str = None) -> list[BaseEnvironmentProfile]:
        if self.env_profiles is None:
            self.load_env_profiles(local_path = local_path)

        profiles = []
        for record in self.env_profiles:
            # Create a BaseEnvironmentProfile instance from the record and add it to the list
            env_profile = BaseEnvironmentProfile(
                pk=record.get("pk", ""),
                codename=record.get("codename", ""),
                source=record.get("source", ""),
                scenario=record.get("scenario", ""),
                agent_goals=record.get("agent_goals", []),
                relationship=record.get("relationship", RelationshipType.stranger),
                age_constraint=record.get("age_constraint", ""),
                occupation_constraint=record.get("occupation_constraint", ""),
                agent_constraint=record.get("agent_constraint", [])
            )
            
            profiles.append(env_profile)

        return profiles
    
    # NOTE: currently only support BaseRelationshipProfile as the same of the jsonl file on Hugging Face
    def load_relationship_profiles(self, local_path: str = None):
        if local_path:
            # Load the dataset from local file
            self.relationship_profiles = list
            try:
                with open(local_path, 'r') as f:
                    for line in f:
                        self.relationship_profiles.append(json.loads(line.strip()))
            except FileNotFoundError:
                print(f"File not found: {local_path}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            # Load the dataset from Hugging Face
            self.relationship_profiles = load_dataset(self.database, data_files=self.relationship_profiles_dataset)

    def get_all_relationship_profiles(self, local_path: str = None) -> list[BaseRelationshipProfile]:
        if self.relationship_profiles is None:
            self.load_relationship_profiles(local_path = local_path)

        profiles = []
        for record in self.relationship_profiles:
            # get all ids
            agent_ids = set
            for key, value in record.items():
                if key.startswith("agent_") and key.endswith("_id") and isinstance(value, str):
                    agent_ids.add(value)

            # Create a BaseRelationshipProfile instance from the record and add it to the list
            rel_profile = BaseRelationshipProfile(
                pk=record.get("pk", ""),
                agent_ids=agent_ids,
                default_relationship=record.get("relationship", RelationshipType.stranger),
                scenario_context=record.get("background_story", "")
            )
            
            profiles.append(rel_profile)

        return profiles
    
    def random_sample_agent_profiles(self, n: int):
        all_profiles = self.get_all_agent_profiles()
        return random.sample(all_profiles, n) if n <= len(all_profiles) else all_profiles
    