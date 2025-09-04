from typing import Any

from pydantic import BaseModel, Field


class BaseAgentProfile(BaseModel):
    model_config = {"protected_namespaces": (), "extra": "allow"}

    pk: str | None = Field(default="")
    first_name: str = Field()
    last_name: str = Field()
    age: int = Field(default=0)
    occupation: str = Field(default="")
    gender: str = Field(default="")
    gender_pronoun: str = Field(default="")
    public_info: str = Field(default="")
    big_five: str = Field(default="")
    moral_values: list[str] = Field(default_factory=list)
    schwartz_personal_values: list[str] = Field(default_factory=list)
    personality_and_values: str = Field(default="")
    decision_making_style: str = Field(default="")
    secret: str = Field(default="")
    model_id: str = Field(default="")
    mbti: str = Field(default="")
    speaking_id: int = Field(
        description="The unique ID of the agent, used for specifying speaking order"
    )
    tag: str = Field(
        default="",
        description="The tag of the agent, used for searching, could be convenient to document agent profiles from different works and sources",
    )

    def to_background_string(self, agent_id: int) -> str:
        info_parts = []
        all_fields = self.model_dump()
        skip_fields = {"pk"}

        field_display_names = {
            "first_name": None,
            "last_name": None,
            "age": "Age",
            "occupation": "Occupation",
            "gender": "Gender",
            "gender_pronoun": "Gender Pronoun",
            "public_info": "Public Info",
            "big_five": "Big Five",
            "moral_values": "Moral Values",
            "schwartz_personal_values": "Schwartz Values",
            "personality_and_values": "Personality",
            "decision_making_style": "Decision Making Style",
            "secret": "Secret",
            "model_id": "Model ID",
            "mbti": "MBTI",
            "tag": "Tag",
        }

        if all_fields.get("first_name") or all_fields.get("last_name"):
            name = f"{all_fields.get('first_name', '')} {all_fields.get('last_name', '')}".strip()
            info_parts.append(f"Name: {name}")

        for field_name, field_value in all_fields.items():
            if (
                field_name in skip_fields
                or field_name in ["first_name", "last_name"]
                or not field_value
                or field_value == ""
                or field_value == []
                or field_value == 0
            ):
                continue

            if isinstance(field_value, list):
                field_value = ", ".join(str(v) for v in field_value)

            display_name = field_display_names.get(
                field_name, field_name.replace("_", " ").title()
            )
            info_parts.append(f"{display_name}: {field_value}")

        background_text = "; ".join(info_parts)
        return f"<root><p viewer='agent_{agent_id}'>{background_text}</p></root>"

    def add_field(self, field_name: str, field_value: Any) -> None:
        setattr(self, field_name, field_value)

    def remove_field(self, field_name: str) -> bool:
        if hasattr(self, field_name):
            if field_name in self.__class__.model_fields:
                if field_name == "age":
                    setattr(self, field_name, 0)
                elif field_name in [
                    "moral_values",
                    "schwartz_personal_values",
                    "agent_goals",
                ]:
                    setattr(self, field_name, [])
                else:
                    setattr(self, field_name, "")
            else:
                delattr(self, field_name)
            return True
        return False
