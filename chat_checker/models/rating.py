from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional
from pydantic import BaseModel, Field


class DimensionType(StrEnum):
    CONVERSATIONAL = "conversational"
    TASK_ORIENTED = "task-oriented"
    GENERAL = "general"


@dataclass
class RatingDimension:
    key: str
    title: str
    rating_question: str
    type: DimensionType


class KeyedDimensionRating(BaseModel):
    key: str = Field(..., description="The key for the dimension")
    reasoning: str = Field(..., description="The reasoning for the rating")
    rating: int = Field(..., description="The rating for the dimension from 1 to 5")


class DialogueRating(BaseModel):
    dimension_ratings: list[KeyedDimensionRating] = Field(
        ...,
        description="A list of dimension ratings with one rating per requested dimension",
    )


@dataclass
class RatingScale:
    min: int
    max: int


@dataclass
class RatingDimensionAnnotation:
    ratings: list[int]
    avg_rating: Optional[float]
    scale: RatingScale = field(default_factory=lambda: RatingScale(min=1, max=5))

    @property
    def mode_rating(self) -> Optional[int]:
        if not self.ratings:
            return None
        return max(set(self.ratings), key=self.ratings.count)


class DialogueDimensionRating(BaseModel):
    reasoning: str = Field(..., description="The short reasoning for the rating")
    rating: int = Field(..., description="The rating for the dimension from 1 to 5")
