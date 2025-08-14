from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5


class BaseRelationshipProfile(BaseModel):
    """Base relationship profile that supports setting a uniform relationship type among all agents"""

    model_config = {"extra": "allow"}

    pk: str | None = Field(default="")

    # Agent list
    agent_ids: list[str] = Field(
        default_factory=list,
        description="List of all agent IDs participating in the relationship network",
    )

    # Unified relationship type
    default_relationship: RelationshipType = Field(
        default=RelationshipType.stranger,
        description="Unified relationship type among all agents: 0=stranger, 1=know_by_name, 2=acquaintance, 3=friend, 4=romantic_relationship, 5=family_member",
    )

    # Scenario description
    scenario_context: str | None = Field(
        default=None,
        description="Scenario background description of the relationship network",
    )

    tag: str = Field(
        default="",
        description="Tag for the relationship network, used for classification and search",
    )

    def add_agents(self, agent_ids: list[str]):
        """Add agents to the relationship network"""
        for agent_id in agent_ids:
            if agent_id not in self.agent_ids:
                self.agent_ids.append(agent_id)

    def get_relationship(self, agent1_id: str, agent2_id: str) -> RelationshipType:
        """Get relationship between two agents (base version returns uniform relationship)"""
        if agent1_id == agent2_id:
            return RelationshipType.family_member  # Self-relationship

        if agent1_id in self.agent_ids and agent2_id in self.agent_ids:
            return self.default_relationship
        else:
            return RelationshipType.stranger  # Default for agents not in network

    def get_agent_relationships(self, agent_id: str) -> dict[str, RelationshipType]:
        """Get relationships between a specific agent and all other agents"""
        relationships = {}
        for other_agent_id in self.agent_ids:
            if other_agent_id != agent_id:
                relationships[other_agent_id] = self.get_relationship(
                    agent_id, other_agent_id
                )
        return relationships

    def to_natural_language(self) -> str:
        """Convert relationship network to natural language description"""
        descriptions = []

        if self.scenario_context:
            descriptions.append(f"Scenario: {self.scenario_context}")

        descriptions.append(f"Participants: {', '.join(self.agent_ids)}")

        rel_name = {
            RelationshipType.stranger: "strangers",
            RelationshipType.know_by_name: "know by name",
            RelationshipType.acquaintance: "acquaintances",
            RelationshipType.friend: "friends",
            RelationshipType.romantic_relationship: "romantic partners",
            RelationshipType.family_member: "family members",
        }.get(self.default_relationship, "unknown relationship")

        descriptions.append(f"All participants are {rel_name}")

        return "\n".join(descriptions)


class FineGrainedRelationshipProfile(BaseRelationshipProfile):
    """Multi-agent relationship profile that inherits from base class and supports personalized pairwise relationships"""

    # Pairwise relationship mapping - uses ordered agent pairs as keys, overrides default relationships
    pairwise_relationships: dict[str, RelationshipType] = Field(
        default_factory=dict,
        description="Personalized pairwise relationships between agents, key format: 'agent1_id:agent2_id' (sorted lexicographically)",
    )

    # Relationship background stories
    relationship_stories: dict[str, str] = Field(
        default_factory=dict,
        description="Background stories for each relationship pair, same key format as pairwise_relationships",
    )

    def _get_relationship_key(self, agent1_id: str, agent2_id: str) -> str:
        """Generate standardized relationship key name (sorted lexicographically)"""
        if agent1_id == agent2_id:
            raise ValueError("Cannot create relationship between agent and itself")
        return f"{min(agent1_id, agent2_id)}:{max(agent1_id, agent2_id)}"

    def set_relationship(
        self,
        agent1_id: str,
        agent2_id: str,
        relationship: RelationshipType,
        background_story: str = None,
    ):
        """Set personalized relationship between two agents"""
        # Ensure agents are in the list
        if agent1_id not in self.agent_ids:
            self.agent_ids.append(agent1_id)
        if agent2_id not in self.agent_ids:
            self.agent_ids.append(agent2_id)

        key = self._get_relationship_key(agent1_id, agent2_id)
        self.pairwise_relationships[key] = relationship

        if background_story:
            self.relationship_stories[key] = background_story

    def get_relationship(self, agent1_id: str, agent2_id: str) -> RelationshipType:
        """Get relationship between two agents (overrides parent method, prioritizes personalized relationships)"""
        if agent1_id == agent2_id:
            return RelationshipType.family_member  # Self-relationship

        # Check if there's a personalized relationship setting
        key = self._get_relationship_key(agent1_id, agent2_id)
        if key in self.pairwise_relationships:
            return self.pairwise_relationships[key]

        # Otherwise return default relationship
        return super().get_relationship(agent1_id, agent2_id)

    def get_relationship_story(self, agent1_id: str, agent2_id: str) -> str | None:
        """Get relationship background story between two agents"""
        if agent1_id == agent2_id:
            return None

        key = self._get_relationship_key(agent1_id, agent2_id)
        return self.relationship_stories.get(key)

    def to_natural_language(self) -> str:
        """Convert relationship network to natural language description (overrides parent method, shows personalized relationships)"""
        descriptions = []

        if self.scenario_context:
            descriptions.append(f"Scenario: {self.scenario_context}")

        descriptions.append(f"Participants: {', '.join(self.agent_ids)}")

        # Show default relationship
        default_rel_name = {
            RelationshipType.stranger: "strangers",
            RelationshipType.know_by_name: "know by name",
            RelationshipType.acquaintance: "acquaintances",
            RelationshipType.friend: "friends",
            RelationshipType.romantic_relationship: "romantic partners",
            RelationshipType.family_member: "family members",
        }.get(self.default_relationship, "unknown relationship")

        descriptions.append(f"Default relationship: {default_rel_name}")

        # Show personalized relationships
        if self.pairwise_relationships:
            descriptions.append("Personalized relationships:")
            for agent1_id in self.agent_ids:
                for agent2_id in self.agent_ids:
                    if agent1_id < agent2_id:  # Avoid duplicates
                        key = self._get_relationship_key(agent1_id, agent2_id)
                        if key in self.pairwise_relationships:
                            relationship = self.pairwise_relationships[key]
                            story = self.relationship_stories.get(key)

                            rel_name = {
                                RelationshipType.stranger: "strangers",
                                RelationshipType.know_by_name: "know by name",
                                RelationshipType.acquaintance: "acquaintances",
                                RelationshipType.friend: "friends",
                                RelationshipType.romantic_relationship: "romantic partners",
                                RelationshipType.family_member: "family members",
                            }.get(relationship, "unknown relationship")

                            rel_desc = f"  {agent1_id} <-> {agent2_id}: {rel_name}"
                            if story:
                                rel_desc += f" ({story})"
                            descriptions.append(rel_desc)

        return "\n".join(descriptions)

    def get_relationship_matrix(self) -> dict[str, dict[str, RelationshipType]]:
        """Get relationship matrix for visualization and analysis"""
        matrix = {}
        for agent1 in self.agent_ids:
            matrix[agent1] = {}
            for agent2 in self.agent_ids:
                matrix[agent1][agent2] = self.get_relationship(agent1, agent2)
        return matrix
