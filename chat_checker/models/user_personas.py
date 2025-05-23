from enum import StrEnum
from typing import Union

from pydantic import BaseModel, Field


class PersonaType(StrEnum):
    STANDARD = "standard"
    CHALLENGING = "challenging"
    ADVERSARIAL = "adversarial"


class Persona(BaseModel):
    persona_id: str = Field(..., description="The ID of the persona")
    type: PersonaType = Field(..., description="The type of the persona")
    profile: Union[str, dict] = Field(
        ..., description="A structured description of the persona"
    )
    task: str = Field(..., description="The task the persona has for the dialogue")
    generated: bool = Field(
        True,
        description="Whether the persona was generated by the system or not",
    )


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class PersonalityTraitExpression(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OCEANPersonality(BaseModel):
    openness: PersonalityTraitExpression = Field(
        ..., description="The openness of the persona"
    )
    conscientiousness: PersonalityTraitExpression = Field(
        ..., description="The conscientiousness of the persona"
    )
    extraversion: PersonalityTraitExpression = Field(
        ..., description="The extraversion of the persona"
    )
    agreeableness: PersonalityTraitExpression = Field(
        ..., description="The agreeableness of the persona"
    )
    neuroticism: PersonalityTraitExpression = Field(
        ..., description="The neuroticism of the persona"
    )


class OurGeneratedPersona(BaseModel):
    number: int = Field(..., description="The number of the persona")
    name: str = Field(..., description="The name of the persona")
    gender: Gender = Field(..., description="The gender of the persona")
    age: int = Field(..., description="The age of the persona")
    background_info: list[str] = Field(
        ...,
        description="The background information of the persona. This is a concise background information that an actor must know about the persona for interacting with the given chatbot",
    )
    personality: OCEANPersonality = Field(
        ..., description="The personality of the persona"
    )
    interaction_style: list[str] = Field(
        ...,
        description="The interaction/writing style of the persona reflecting the personality. The interaction style must guide an LLM to write as human-like as possible while playing the persona in a conversation with the chatbot.",
    )
    task: str = Field(
        ...,
        description="The task the persona has for the dialogue. The task description should be specific and brief.",
    )


class GeneratedPersonas(BaseModel):
    personas: list[OurGeneratedPersona] = Field(
        ..., description="The list of generated personas"
    )
