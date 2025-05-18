import os

from litellm import (
    supports_response_schema,
    get_supported_openai_params,
    completion_cost,
)
from litellm.types.utils import ModelResponse

from chat_checker.models.llm import UsageCost

DEFAULT_LLM = os.getenv("CHAT_CHECKER_DEFAULT_LLM", "gpt-4o-2024-08-06")


def supports_structured_outputs(model: str) -> bool:
    """
    Check if the model supports structured outputs.
    Args:
        model (str): The name of the model.
    Returns:
        bool: True if the model supports structured outputs, False otherwise.
    """
    supports_response_format = "response_format" in (
        get_supported_openai_params(model) or []
    )
    return supports_response_format and supports_response_schema(model)


def compute_total_usage(generations: list[ModelResponse]) -> UsageCost:
    # ModelResponse objects do have the usage attribute (https://docs.litellm.ai/docs/completion/output) it is just not typed in the stub
    prompt_tokens = sum([gen.usage.prompt_tokens for gen in generations])  # type: ignore
    completion_tokens = sum([gen.usage.completion_tokens for gen in generations])  # type: ignore
    total_tokens = sum([gen.usage.total_tokens for gen in generations])  # type: ignore
    total_cost = sum([completion_cost(gen) for gen in generations])
    return UsageCost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=total_cost,
    )
