from pathlib import Path
from typing import List, Optional

from openai.types.chat import ChatCompletionMessageParam
from litellm import (
    completion,
)
from litellm.types.utils import ModelResponse, Choices

from chat_checker.models.breakdowns import BreakdownDescription
from chat_checker.models.chatbot import ChatbotInfo
from chat_checker.models.dialogue import DialogueTurn
from chat_checker.user_simulation.prompt_components import (
    END_CONVERSATION_INSTRUCTION,
    MAX_TURN_LENGTH_CONSTRAINT,
    SPECIFIC_LENGTH_GUIDANCE,
)
from chat_checker.user_simulation.user_simulator_base import (
    OurUserSimulatorBase,
    UserSimulatorBase,
    UserSimulatorResponse,
)
from chat_checker.utils.llm_utils import DEFAULT_LLM
from chat_checker.utils.misc_utils import get_matching_api_key
from chat_checker.utils.prompt_utils import (
    generate_chat_history_str,
)
from chat_checker.user_simulation.test_user_simulator.test_user_simulator_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)

BASE_DIR = Path(__file__).parent


class TestUserSimulator(OurUserSimulatorBase):
    def __init__(
        self,
        target_breakdown: BreakdownDescription,
        chatbot_info: ChatbotInfo,
        model: str = DEFAULT_LLM,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        typical_user_turn_length: Optional[str] = None,
        max_user_turn_length: Optional[str] = None,
    ):
        # Note: by default temperature is left at the default for more diverse responses
        super().__init__(
            chatbot_info,
            model,
            temperature,
            seed,
            typical_user_turn_length,
            max_user_turn_length,
        )
        self.target_breakdown = target_breakdown

    def generate_response(
        self, chat_history: List[DialogueTurn]
    ) -> UserSimulatorResponse:
        if self.typical_user_turn_length is not None:
            specific_length_guidance = SPECIFIC_LENGTH_GUIDANCE.format(
                typical_user_turn_length=self.typical_user_turn_length
            )
        else:
            specific_length_guidance = ""

        if self.max_user_turn_length is not None:
            max_turn_length_constraint = MAX_TURN_LENGTH_CONSTRAINT.format(
                max_turn_length=self.max_user_turn_length
            )
        else:
            max_turn_length_constraint = ""

        system_prompt = SYSTEM_PROMPT.format(
            error_type=self.target_breakdown.title,
            chatbot_info=self.chatbot_info.dump_as_yaml_without_task(),
            tester_instructions=self.target_breakdown.tester_instructions,
            specific_length_guidance=specific_length_guidance,
            max_turn_length_constraint=max_turn_length_constraint,
            end_conversation_instruction=END_CONVERSATION_INSTRUCTION,
        )

        chat_history_str = generate_chat_history_str(
            chat_history,
            user_tag="YOU",
            chatbot_tag="CHATBOT",
        )
        turn_number = len(chat_history) + 1

        user_prompt = USER_PROMPT.format(
            chat_history_str=chat_history_str,
            turn_number=turn_number,
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response: ModelResponse = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            api_key=get_matching_api_key(self.model).get_secret_value(),
            drop_params=True,
        )

        # for type-checking
        assert isinstance(response, ModelResponse)
        assert isinstance(response.choices[0], Choices)
        if not response.choices[0].message.content:
            raise ValueError("Missing simulator response")

        answer = response.choices[0].message.content
        if answer is not None:
            answer = answer.strip("'").strip('"')
            cleaned_answer, conversation_over = (
                UserSimulatorBase.handle_model_response_end(answer)
            )
        else:
            cleaned_answer = ""
            conversation_over = True

        return UserSimulatorResponse(
            response_message=cleaned_answer,
            is_end=conversation_over,
            prompt_messages=messages,
            model_response=response,
        )
