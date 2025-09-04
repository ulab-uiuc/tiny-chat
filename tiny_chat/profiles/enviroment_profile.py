from pydantic import BaseModel, Field

from .relationship_profile import RelationshipType


class BaseEnvironmentProfile(BaseModel):
    model_config = {"extra": "allow"}
    pk: str | None = Field(default="")
    codename: str = Field(
        default="",
        description="The codename of the environment",
    )
    source: str = Field(
        default="",
        description="The source of the environment",
    )
    scenario: str = Field(
        description="A concrete scenario of where the social interaction takes place, the scenario should have two agents (agent1 and agent2), and you should illustrate the relationship between the two agents, and for what purpose agent1 is interacting with agent2. Please avoid mentioning specific names and occupations in the scenario and keep all the mentions gender-neutral. Also avoid generating scenarios that requires childrend (below 18) or elderly (above 70) to be involved.",
    )
    agent_goals: list[str] = Field(
        default_factory=list,
        description="The social goals of each agent, which could include <extra_info>...</extra_info>, <clarification_hint>...</clarification_hint>, and <strategy_hint>...</strategy_hint> to help the agent achieve the goal. Avoid providing too specific strategy hint, try to be as abstract as possible. For example, use 'you can provide financial benefits to achieve your goal' instead of 'you can buy him a boba tea to achieve your goal.'",
    )
    relationship: RelationshipType = Field(
        default=RelationshipType.stranger,
        description="The relationship between the two agents, choose from: stranger, know_by_name, acquaintance, friend, romantic_relationship, family_member. Do not make up a relationship, but choose from the list, 0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member",
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
        default="",
        description="The tag of the environment, used for searching, could be convenient to document environment profiles from different works and sources",
    )
