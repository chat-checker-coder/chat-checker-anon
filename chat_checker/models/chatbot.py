from enum import StrEnum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml

from chat_checker.dialogue_rating.rating_dimensions import (
    DEFAULT_CONVERSATIONAL_DIMENSIONS,
    DEFAULT_TASK_ORIENTED_DIMENSIONS,
)
from chat_checker.models.rating import RatingDimension


class ChatbotType(StrEnum):
    TASK_ORIENTED = "task-oriented"
    CONVERSATIONAL = "conversational"


class ChatbotInfo(BaseModel):
    name: str = Field(..., description="The chatbot name")
    description: str = Field(..., description="A brief description")
    type: ChatbotType = Field(
        default=ChatbotType.TASK_ORIENTED, description="The chatbot type"
    )
    interaction_method: Optional[str] = Field(
        None, description="The chatbot interaction method (e.g., text-based)"
    )
    task: Optional[str] = Field(
        None, description="The task for the chatbot in the dialogue"
    )
    constraints: Optional[list[str]] = Field(
        None,
        description="Constraints for the expected behavior of the chatbot. This can be used for specifying behavior for out-of-domain requests.",
    )
    known_limitations: Optional[list[str]] = Field(
        None,
        description="Known limitations of the chatbot. This can be used for specifying things the chatbot is not expected to do.",
    )
    available_languages: list[str] = Field(
        ..., description="The languages the chatbot is available in"
    )

    def __str__(self) -> str:
        return yaml.safe_dump(
            self.model_dump(), indent=4, sort_keys=False, allow_unicode=True
        )

    def dump_as_yaml_without_task(self) -> str:
        return yaml.safe_dump(
            self.model_dump(exclude={"task"}),
            indent=4,
            sort_keys=False,
            allow_unicode=True,
        )


class UserSimulationConfig(BaseModel):
    max_user_turns: int = Field(
        ..., description="The maximum number of user turns in a dialogue"
    )
    typical_user_turn_length: Optional[str] = Field(
        None, description="The typical user turn length expressed in words"
    )
    max_user_turn_length: Optional[str] = Field(
        None, description="The maximum user turn length expressed in words"
    )


class Chatbot(BaseModel):
    base_directory: Path = Field(
        ...,
        description="The base directory of the chatbot configuration, client implementation, and runs",
        exclude=True,
    )

    id: str = Field(..., description="The chatbot ID")
    info: ChatbotInfo = Field(
        ..., description="The chatbot information", validation_alias="chatbot_info"
    )
    user_simulation_config: UserSimulationConfig = Field(
        ..., description="The user simulation configuration for the chatbot"
    )

    @property
    def rating_dimensions(self) -> list[RatingDimension]:
        if self.info.type == ChatbotType.TASK_ORIENTED:
            return DEFAULT_TASK_ORIENTED_DIMENSIONS
        elif self.info.type == ChatbotType.CONVERSATIONAL:
            return DEFAULT_CONVERSATIONAL_DIMENSIONS
