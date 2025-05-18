from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import yaml
from pydantic import BaseModel, Field


@dataclass
class BreakdownDescription:
    title: str
    description: str
    example: Optional[str]
    tester_instructions: str


class BreakdownDecision(StrEnum):
    BREAKDOWN = "breakdown"
    NO_BREAKDOWN = "no_breakdown"


class BreakdownAnnotation(BaseModel):
    reasoning: str = Field(
        ..., description="The reason for the decision and classification."
    )
    score: float = Field(
        ...,
        description="The score for the chatbot's response. 0 indicates a complete breakdown and 1 indicates a seamless conversation.",
    )
    decision: BreakdownDecision = Field(
        ..., description="The decision for the breakdown annotation."
    )
    breakdown_types: list[str] = Field(
        ...,
        description="All fitting breakdown types that occurred in the turn. Empty if no breakdown was detected.",
    )


if __name__ == "__main__":
    dummy_bd_annotation = BreakdownAnnotation(
        reasoning="The chatbot failed to provide the requested information",
        score=0.2,
        decision=BreakdownDecision.BREAKDOWN,
        breakdown_types=["task_oriented.task_success_failures"],
    )
    print(yaml.safe_dump({"decision": BreakdownDecision.BREAKDOWN}))
    print(
        yaml.safe_dump(
            dummy_bd_annotation.model_dump(), default_flow_style=False, sort_keys=False
        )
    )
