from pydantic import BaseModel


class UsageCost(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
