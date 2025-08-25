from datasets import load_dataset
from tiny_chat.profiles.agent_profile import BaseAgentProfile
import random


class dataLoader:
    """a class to load data from hugging face"""

    def __init__(self, database_name: str = "skyyyyks/tiny-chat"):
        self.database = database_name
        self.agent_profiles_dataset = "agent_profiles.jsonl"
        self.env_profiles_dataset = "environment_profiles.jsonl"
        self.relation_profiles_dataset = "relationship_profiles.jsonl"
        self.agent_profiles = None
        self.env_profiles = None
        self.relation_profiles = None

    def load_agent_profiles(self):
        # Load the dataset from Hugging Face
        self.agaent_profiles = load_dataset(self.database, data_files=self.agent_profiles_dataset)

    def get_all_agent_profiles(self):
        if self.agaent_profiles is None:
            self.load_agent_profiles()

        profiles = []
        for record in self.agaent_profiles:
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
    
    def random_sample_agent_profiles(self, n: int):
        all_profiles = self.get_all_agent_profiles()
        return random.sample(all_profiles, n) if n <= len(all_profiles) else all_profiles
    