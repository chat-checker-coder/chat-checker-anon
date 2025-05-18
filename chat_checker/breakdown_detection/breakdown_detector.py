from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import json
from tqdm import tqdm

from openai.types.chat import ChatCompletionMessageParam
from litellm import (
    get_supported_openai_params,
    completion,
)
from litellm.types.utils import ModelResponse, Choices

from chat_checker.breakdown_detection.breakdown_taxonomy import (
    get_flattened_taxonomy,
)
from chat_checker.models.breakdowns import BreakdownAnnotation, BreakdownDecision
from chat_checker.models.chatbot import ChatbotInfo
from chat_checker.models.dialogue import DialogueTurn, SpeakerRole
from chat_checker.utils.llm_utils import DEFAULT_LLM, supports_structured_outputs
from chat_checker.utils.misc_utils import get_matching_api_key
from chat_checker.utils.prompt_utils import (
    generate_chat_history_str,
    generate_ghassel_chat_history_str,
)
from chat_checker.breakdown_detection.breakdown_detection_prompts import (
    taxonomy_item_str,
    chatbot_info_description_str,
    output_format_str,
    breakdown_identification_system_prompt,
    breakdown_identification_user_prompt,
    ghassel_breakdown_definition,
    ghassel_output_format,
    ghassel_breakdown_detection_prompt,
)

# Build the path to the .env file
BASE_DIR = Path(__file__).parent
env_path = BASE_DIR.parent / ".env"
# Load the environment variables from the .env file
load_dotenv(env_path, override=True)


class BreakdownIdentifier(ABC):
    @abstractmethod
    def identify_breakdowns(
        self,
        chat_history: list[DialogueTurn],
        last_bot_utterance: str,
        is_task_oriented: bool = True,
        chatbot_info: Optional[ChatbotInfo] = None,
        llm_name: str = DEFAULT_LLM,
        seed: Optional[int] = 42,
    ) -> Tuple[BreakdownAnnotation, List[ChatCompletionMessageParam], ModelResponse]:
        pass


class OurBreakdownIdentifier(BreakdownIdentifier):
    def identify_breakdowns(
        self,
        chat_history: list[DialogueTurn],
        last_bot_utterance: str,
        is_task_oriented: bool = True,
        chatbot_info: Optional[ChatbotInfo] = None,
        llm_name: str = DEFAULT_LLM,
        seed: Optional[int] = 42,
    ) -> Tuple[BreakdownAnnotation, List[ChatCompletionMessageParam], ModelResponse]:
        use_structured_outputs = True
        output_format = ""  # By default, we use the structured output mode with the BreakdownAnnotation class
        if not supports_structured_outputs(llm_name):
            # Make sure the model at least supports json mode
            assert "response_format" in (get_supported_openai_params(llm_name) or [])
            use_structured_outputs = False
            output_format = output_format_str

        breakdowns_with_descriptions = get_flattened_taxonomy(is_task_oriented)
        breakdown_taxonomy_str = "\n".join(
            [
                taxonomy_item_str.format(
                    breakdown_name=breakdown.title,
                    breakdown_description=breakdown.description,
                )
                for breakdown in breakdowns_with_descriptions.values()
            ]
        )

        chatbot_info_desc = ""
        if chatbot_info:
            chatbot_info_desc = chatbot_info_description_str.format(
                chatbot_info=chatbot_info
            )

        system_prompt = breakdown_identification_system_prompt.format(
            breakdown_taxonomy=breakdown_taxonomy_str,
            chatbot_info_desc=chatbot_info_desc,
            output_format=output_format,
        )

        dialogue_str = generate_chat_history_str(chat_history, "User", "Chatbot")

        latest_bot_utterance_str = (
            f'{len(chat_history) + 1}. Chatbot: "{last_bot_utterance}"'
        )

        user_prompt = breakdown_identification_user_prompt.format(
            chat_history_str=dialogue_str,
            last_bot_utterance=latest_bot_utterance_str,
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        identification_response: ModelResponse = completion(
            model=llm_name,
            temperature=0,
            seed=seed,
            messages=messages,
            response_format=BreakdownAnnotation
            if use_structured_outputs
            else {"type": "json_object"},
            api_key=get_matching_api_key(llm_name).get_secret_value(),
            drop_params=True,  # drop all params that are not supported by the model (e.g., temperature 0 is not supported by o-series models)
        )
        # for type-checking
        assert isinstance(identification_response, ModelResponse)
        assert isinstance(identification_response.choices[0], Choices)
        if not identification_response.choices[0].message.content:
            raise ValueError("Missing breakdown classification")

        breakdown_annotation_json = json.loads(
            identification_response.choices[0].message.content
        )
        breakdown_annotation = BreakdownAnnotation(**breakdown_annotation_json)

        return breakdown_annotation, messages, identification_response


class GhasselBreakdownIdentifier(BreakdownIdentifier):
    def __init__(self, use_breakdown_taxonomy=False):
        self.use_breakdown_taxonomy = use_breakdown_taxonomy
        super().__init__()

    def identify_breakdowns(
        self,
        chat_history: list[DialogueTurn],
        last_bot_utterance: str,
        is_task_oriented: bool = True,
        chatbot_info: Optional[ChatbotInfo] = None,
        llm_name: str = DEFAULT_LLM,
        seed: Optional[int] = 42,
    ) -> Tuple[BreakdownAnnotation, List[ChatCompletionMessageParam], ModelResponse]:
        # Make sure the model supports json mode
        assert "response_format" in (get_supported_openai_params(llm_name) or [])
        # Adapted from paper "Are Large Language Models General-Purpose Solvers for Dialogue Breakdown Detection? An Empirical Investigation" (https://ieeexplore.ieee.org/document/10667232)
        breakdown_definition = ""
        if self.use_breakdown_taxonomy:
            # Using my own breakdown taxonomy
            breakdown_taxonomy_header = "## Breakdown Taxonomy"
            breakdown_taxonomy_str = "When evaluating the chatbot's response, consider the following breakdown types, which represent common disruptions:\n"
            breakdowns_with_descriptions = get_flattened_taxonomy(is_task_oriented)
            breakdown_taxonomy_str += "\n".join(
                [
                    taxonomy_item_str.format(
                        breakdown_name=breakdown.title,
                        breakdown_description=breakdown.description,
                    )
                    for breakdown in breakdowns_with_descriptions.values()
                ]
            )
            breakdown_definition = ghassel_breakdown_definition + "\n\n"
            breakdown_definition += (
                f"{breakdown_taxonomy_header}\n{breakdown_taxonomy_str}"
            )
        else:
            # From paper
            breakdown_definition = ghassel_breakdown_definition

        chat_history_str = generate_ghassel_chat_history_str(chat_history)

        latest_bot_utterance_str = f"{len(chat_history) + 1}. Bot: {last_bot_utterance}"

        prompt = ghassel_breakdown_detection_prompt.format(
            breakdown_definition=breakdown_definition,
            chat_history_str=chat_history_str,
            last_bot_utterance=latest_bot_utterance_str,
            output_format=ghassel_output_format,
        )

        # Note: we use system prompt following https://github.com/aghassel/LLM-dialogue-breakdown-detection-challenge/blob/main/Analysis/openai_async.py#L56
        message_role = (
            "system" if not llm_name.startswith("gemini/") else "user"
        )  # Gemini requires a user role message to be present (https://github.com/BerriAI/litellm/issues/8467)
        messages: List[ChatCompletionMessageParam] = [
            {"role": message_role, "content": prompt},
        ]

        response_format: Optional[dict[str, str]] = {"type": "json_object"}
        if llm_name == "gpt-4" or llm_name == "gpt-4-0613":
            # 'response_format' of type 'json_object' is not supported with this model
            response_format = None

        detection_response: ModelResponse = completion(
            model=llm_name,
            temperature=0,
            seed=seed,
            response_format=response_format,
            messages=messages,
            api_key=get_matching_api_key(llm_name).get_secret_value(),
            drop_params=True,  # drop all params that are not supported by the model (e.g., temperature 0 is not supported by o-series models)
        )
        # for type-checking
        assert isinstance(detection_response, ModelResponse)
        assert isinstance(detection_response.choices[0], Choices)

        classification = detection_response.choices[0].message.content
        if classification is None:
            raise ValueError("Missing breakdown classification")
        classification_dict = json.loads(classification)
        if type(classification_dict) is list:
            # Needed because Ghassel prompts the model to return a list of JSON objects even though only one breakdown is analyzed
            classification_dict = classification_dict[0]
        classification_dict["breakdown_types"] = []
        classification_dict["decision"] = (
            BreakdownDecision.BREAKDOWN
            if classification_dict["decision"] == "BREAKDOWN"
            else BreakdownDecision.NO_BREAKDOWN
        )
        breakdown_annotation = BreakdownAnnotation(**classification_dict)
        return breakdown_annotation, messages, detection_response


def find_dialogue_breakdowns(
    chat_history: list[DialogueTurn],
    is_task_oriented: bool = True,
    chatbot_info: Optional[ChatbotInfo] = None,
    breakdown_identifier: BreakdownIdentifier = OurBreakdownIdentifier(),
    save_prompts=False,
    save_dir="./prompts/breakdown_detection",
    breakdown_detector_model: str = DEFAULT_LLM,
    seed: Optional[int] = None,
) -> list[ModelResponse]:
    model_responses = []
    for i, turn in tqdm(
        enumerate(chat_history),
        desc="Finding dialogue breakdowns",
        total=len(chat_history),
    ):
        if turn.role == SpeakerRole.DIALOGUE_SYSTEM:
            conversation_history = chat_history[:i]
            last_bot_utterance = turn.content
            if last_bot_utterance != "chatbot_error":
                breakdown_info, prompt, model_response = (
                    breakdown_identifier.identify_breakdowns(
                        conversation_history,
                        last_bot_utterance,
                        is_task_oriented,
                        chatbot_info,
                        breakdown_detector_model,
                        seed=seed,
                    )
                )
                turn.breakdown_annotation = breakdown_info
                model_responses.append(model_response)

                prompt_str = "\n\n".join(
                    [f"{message['role']}: {message['content']}" for message in prompt]
                )
                os.makedirs(save_dir, exist_ok=True)
                if save_prompts:
                    with open(
                        f"{save_dir}/turn_{i + 1}_prompt.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(prompt_str)
    return model_responses
