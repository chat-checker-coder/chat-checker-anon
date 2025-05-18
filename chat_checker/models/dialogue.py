from enum import StrEnum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from chat_checker.models.breakdowns import BreakdownAnnotation
from chat_checker.models.rating import (
    DialogueDimensionRating,
    RatingDimensionAnnotation,
)


class SpeakerRole(StrEnum):
    USER = "user"
    DIALOGUE_SYSTEM = "dialogue_system"


class DialogueTurn(BaseModel):
    turn_id: int = Field(..., description="The ID of the turn")
    role: SpeakerRole = Field(..., description="The role of the speaker of the turn")
    content: str = Field(..., description="The content of the turn")
    breakdown_annotation: Optional[BreakdownAnnotation] = Field(
        None, description="The breakdown annotation of the turn"
    )


class FinishReason(StrEnum):
    MAX_TURNS_REACHED = "max_turns_reached"
    USER_ENDED = "user_ended_chat"
    CHATBOT_ENDED = "chatbot_ended_chat"
    USER_SIMULATOR_ERROR = "user_simulator_error"
    CHATBOT_ERROR = "chatbot_error"


class Dialogue(BaseModel):
    dialogue_id: str = Field(..., description="The ID of the dialogue")
    path: Path = Field(..., description="The path to the dialogue file", exclude=True)
    user_name: str = Field(..., description="The name of the user who led the dialogue")
    chat_history: list[DialogueTurn] = Field(
        ..., description="The turns of the dialogue"
    )
    finish_reason: FinishReason = Field(
        ..., description="The reason the dialogue finished"
    )
    error: Optional[str] = Field(
        None, description="The error that caused the dialogue to finish"
    )
    ratings: Optional[dict[str, DialogueDimensionRating]] = Field(
        None,
        description="The computed ratings of the dialogue across different dimensions",
    )
    human_rating_annotations: Optional[dict[str, RatingDimensionAnnotation]] = Field(
        None,
        description="The human ratings of the dialogue across different dimensions",
    )
    chat_statistics: Optional[dict] = Field(
        None, description="The chat statistics of the dialogue"
    )
    simulation_cost_statistics: Optional[dict] = Field(
        None, description="The simulation cost statistics of the dialogue"
    )
    breakdown_stats: Optional[dict] = Field(
        None, description="The breakdown statistics of the dialogue"
    )
    eval_stats: Optional[dict] = Field(
        None, description="The evaluation statistics of the dialogue"
    )
