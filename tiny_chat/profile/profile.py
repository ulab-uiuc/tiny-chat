from enum import IntEnum

from pydantic import BaseModel, Field


class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5


class BaseAgentProfile(BaseModel):
    pk: str | None = Field(default='')
    first_name: str = Field()
    last_name: str = Field()
    age: int = Field(default=0)
    occupation: str = Field(default='')
    gender: str = Field(default='')
    gender_pronoun: str = Field(default='')
    public_info: str = Field(default='')
    big_five: str = Field(default='')
    moral_values: list[str] = Field(default_factory=list)
    schwartz_personal_values: list[str] = Field(default_factory=list)
    personality_and_values: str = Field(default='')
    decision_making_style: str = Field(default='')
    secret: str = Field(default='')
    model_id: str = Field(default='')
    mbti: str = Field(default='')
    tag: str = Field(
        default='',
        description='The tag of the agent, used for searching, could be convenient to document agent profiles from different works and sources',
    )

    def to_background_string(self, agent_id: int) -> str:
        info_parts = []
        all_fields = self.model_dump()
        skip_fields = {'pk'} 

        field_display_names = {
            'first_name': None,
            'last_name': None,
            'age': 'Age',
            'occupation': 'Occupation',
            'gender': 'Gender',
            'gender_pronoun': 'Gender Pronoun',
            'public_info': 'Public Info',
            'big_five': 'Big Five',
            'moral_values': 'Moral Values',
            'schwartz_personal_values': 'Schwartz Values',
            'personality_and_values': 'Personality',
            'decision_making_style': 'Decision Making Style',
            'secret': 'Secret',
            'model_id': 'Model ID',
            'mbti': 'MBTI',
            'tag': 'Tag'
        }

        if all_fields.get('first_name') or all_fields.get('last_name'):
            name = f"{all_fields.get('first_name', '')} {all_fields.get('last_name', '')}".strip()
            info_parts.append(f'Name: {name}')
        
        for field_name, field_value in all_fields.items():
            if (field_name in skip_fields or 
                field_name in ['first_name', 'last_name'] or
                not field_value or field_value == '' or field_value == [] or field_value == 0):
                continue
                
            if isinstance(field_value, list):
                field_value = ', '.join(str(v) for v in field_value)
            
            display_name = field_display_names.get(field_name, field_name.replace('_', ' ').title())
            info_parts.append(f'{display_name}: {field_value}')

        background_text = '; '.join(info_parts)
        return f"<root><p viewer='agent_{agent_id}'>{background_text}</p></root>"


    def add_field(self, field_name: str, field_value: any) -> None:
        setattr(self, field_name, field_value)

    def remove_field(self, field_name: str) -> bool:
        if hasattr(self, field_name):
            if field_name in self.__class__.model_fields:
                field_info = self.__class__.model_fields[field_name]
                if hasattr(field_info, 'default') and field_info.default is not None:
                    setattr(self, field_name, field_info.default)
                elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                    setattr(self, field_name, field_info.default_factory())
                else:
                    if field_name == 'age':
                        setattr(self, field_name, 0)
                    elif isinstance(getattr(self, field_name, ''), list):
                        setattr(self, field_name, [])
                    else:
                        setattr(self, field_name, '')
            else:
                delattr(self, field_name)
            return True
        return False

    class Config:
        extra = "allow"


class BaseEnvironmentProfile(BaseModel):
    pk: str | None = Field(default='')
    codename: str = Field(
        default='',
        description='The codename of the environment',
    )
    source: str = Field(
        default='',
        description='The source of the environment',
    )
    scenario: str = Field(
        description='A concrete scenario of where the social interaction takes place, the scenario should have two agents (agent1 and agent2), and you should illustrate the relationship between the two agents, and for what purpose agent1 is interacting with agent2. Please avoid mentioning specific names and occupations in the scenario and keep all the mentions gender-neutral. Also avoid generating scenarios that requires childrend (below 18) or elderly (above 70) to be involved.',
    )
    agent_goals: list[str] = Field(
        default_factory=list,
        description="The social goals of each agent, which could include <extra_info>...</extra_info>, <clarification_hint>...</clarification_hint>, and <strategy_hint>...</strategy_hint> to help the agent achieve the goal. Avoid providing too specific strategy hint, try to be as abstract as possible. For example, use 'you can provide financial benefits to achieve your goal' instead of 'you can buy him a boba tea to achieve your goal.'",
    )
    relationship: RelationshipType = Field(
        default=RelationshipType.stranger,
        description='The relationship between the two agents, choose from: stranger, know_by_name, acquaintance, friend, romantic_relationship, family_member. Do not make up a relationship, but choose from the list, 0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member',
    )
    age_constraint: str | None = Field(
        default=None,
        description="The age constraint of the environment, a list of tuples, each tuple is a range of age, e.g., '[(18, 25), (30, 40)]' means the environment is only available to agent one between 18 and 25, and agent two between 30 and 40",
    )
    occupation_constraint: str | None = Field(
        default=None,
        description="The occupation constraint of the environment, a list of lists, each list is a list of occupations, e.g., '[['student', 'teacher'], ['doctor', 'nurse']]' means the environment is only available to agent one if agent one is a student or a teacher, and agent two is a doctor or a nurse",
    )
    agent_constraint: list[list[str]] | None = Field(
        default=None,
    )
    tag: str = Field(
        default='',
        description='The tag of the environment, used for searching, could be convenient to document environment profiles from different works and sources',
    )
    class Config:
        extra = "allow"




class BaseRelationshipProfile(BaseModel):
    pk: str | None = Field(default='')
    agent_1_id: str = Field()
    agent_2_id: str = Field()
    relationship: RelationshipType = Field(
        description='0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member',
    )  # this could be improved by limiting str to a relationship Enum
    background_story: str | None = Field(default=None)
    tag: str = Field(
        default='',
        description='The tag of the relationship, used for searching, could be convenient to document relationship profiles from different works and sources',
    )
    class Config:
        extra = "allow"
