import json
from typing import Optional, Tuple

from openai.types.chat import ChatCompletionMessageParam
from litellm import completion
from litellm.types.utils import ModelResponse, Choices

from chat_checker.models.chatbot import ChatbotInfo
from chat_checker.models.dialogue import Dialogue, DialogueTurn
from chat_checker.models.rating import (
    DialogueDimensionRating,
    DialogueRating,
    RatingDimension,
)
from chat_checker.utils.llm_utils import DEFAULT_LLM, supports_structured_outputs
from chat_checker.utils.misc_utils import get_matching_api_key
from chat_checker.utils.prompt_utils import generate_chat_history_str
from chat_checker.dialogue_rating.rating_prompts import (
    chatbot_info_description_str,
    rating_example_str,
    rating_examples_str,
    rating_dimension_str,
    dialogue_rating_system_prompt,
    dialogue_rating_user_prompt,
)


def get_dialogue_rating(
    chat_history: list[DialogueTurn],
    rating_dimensions: list[RatingDimension],
    chatbot_info: Optional[ChatbotInfo] = None,
    examples: list[Dialogue] = [],
    rating_model: str = DEFAULT_LLM,
    seed: Optional[int] = 42,
) -> Tuple[
    dict[str, DialogueDimensionRating], list[ChatCompletionMessageParam], ModelResponse
]:
    assert supports_structured_outputs(rating_model)

    if chatbot_info:
        chatbot_info_desc = chatbot_info_description_str.format(
            chatbot_info=chatbot_info
        )
    else:
        chatbot_info_desc = ""

    examples_str = ""
    for i, example in enumerate(examples):
        if example.human_rating_annotations is None:
            continue
        example_chat_str = generate_chat_history_str(
            example.chat_history, user_tag="User", chatbot_tag="Chatbot"
        )
        avg_user_rating_str = ""
        for key, value in example.human_rating_annotations.items():
            if not value.avg_rating:
                continue
            # rescale the ratings to the 1-5 scale
            avg_rating = (value.avg_rating - value.scale.min) / (
                value.scale.max - value.scale.min
            ) * 4 + 1
            avg_user_rating_str += (
                f'Average human rating for dimension "{key}": {avg_rating:.2f}\n'
            )
        examples_str += rating_example_str.format(
            example_num=i + 1,
            chat_history_str=example_chat_str,
            ratings=avg_user_rating_str,
        )

    if examples_str:
        few_shot_examples = rating_examples_str.format(examples=examples_str)
    else:
        few_shot_examples = ""

    rating_dimensions_str = ""
    for rating_dimension in rating_dimensions:
        rating_dimensions_str += (
            rating_dimension_str.format(
                dimension_name=rating_dimension.title,
                dimension_key=rating_dimension.key,
                rating_question=rating_dimension.rating_question,
            )
            + "\n"
        )

    chat_history_str = generate_chat_history_str(
        chat_history, user_tag="User", chatbot_tag="Chatbot"
    )

    system_prompt = dialogue_rating_system_prompt.format(
        chatbot_info_desc=chatbot_info_desc,
        few_shot_examples=few_shot_examples,
        rating_dimensions=rating_dimensions_str,
    )

    user_prompt = dialogue_rating_user_prompt.format(chat_history_str=chat_history_str)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    rating_response: ModelResponse = completion(
        model=rating_model,
        temperature=0,
        seed=seed,
        messages=messages,
        response_format=DialogueRating,
        api_key=get_matching_api_key(rating_model).get_secret_value(),
        drop_params=True,  # drop all params that are not supported by the model (e.g., temperature 0 is not supported by o-series models)
    )
    # for type-checking
    assert isinstance(rating_response, ModelResponse)
    assert isinstance(rating_response.choices[0], Choices)
    if not rating_response.choices[0].message.content:
        raise ValueError("No rating found in the response")
    rating_json = json.loads(rating_response.choices[0].message.content)
    rating = DialogueRating(**rating_json)

    # Convert list of dimension ratings to dict
    # Note: we cannot use dict in the DialogueRating model directly due to this issue: https://community.openai.com/t/pydantic-with-dict-not-working/1046724
    dimension_ratings_dict = {
        rating.key: DialogueDimensionRating(
            reasoning=rating.reasoning,
            rating=rating.rating,
        )
        for rating in rating.dimension_ratings
    }
    # Make sure that all requested dimensions are present in the rating
    for rating_dimension in rating_dimensions:
        if rating_dimension.key not in dimension_ratings_dict:
            raise ValueError(f"Dimension {rating_dimension.key} not found in rating")

    return dimension_ratings_dict, messages, rating_response
