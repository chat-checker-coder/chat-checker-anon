from typing import List, Optional
import yaml

from openai.types.chat import ChatCompletionMessageParam
from litellm import (
    completion,
)
from litellm.types.utils import ModelResponse, Choices

from chat_checker.user_simulation.prompt_components import (
    END_CONVERSATION_INSTRUCTION,
    GENERAL_LENGTH_GUIDANCE,
    MAX_TURN_LENGTH_CONSTRAINT,
    SPECIFIC_LENGTH_GUIDANCE,
)
from chat_checker.models.chatbot import ChatbotInfo
from chat_checker.models.dialogue import DialogueTurn
from chat_checker.models.user_personas import Persona
from chat_checker.user_simulation.persona_simulator.persona_simulator_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from chat_checker.utils.llm_utils import DEFAULT_LLM
from chat_checker.utils.misc_utils import get_matching_api_key
from chat_checker.utils.prompt_utils import (
    generate_chat_history_str,
)
from chat_checker.user_simulation.user_simulator_base import (
    OurUserSimulatorBase,
    UserSimulatorBase,
    UserSimulatorResponse,
)


class PersonaSimulator(OurUserSimulatorBase):
    def __init__(
        self,
        user_persona: Persona,
        chatbot_info: ChatbotInfo,
        model: str = DEFAULT_LLM,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        typical_user_turn_length: Optional[str] = None,
        max_user_turn_length: Optional[str] = None,
    ):
        super().__init__(
            chatbot_info,
            model,
            temperature,
            seed,
            typical_user_turn_length,
            max_user_turn_length,
        )
        self.user_persona = user_persona

    def generate_response(
        self, chat_history: List[DialogueTurn]
    ) -> UserSimulatorResponse:
        persona_model = {
            "profile": self.user_persona.profile,
            "task": self.user_persona.task,
        }
        persona_str = yaml.safe_dump(
            persona_model, indent=4, sort_keys=False, allow_unicode=True
        )
        if self.typical_user_turn_length is not None:
            length_guidance = SPECIFIC_LENGTH_GUIDANCE.format(
                typical_user_turn_length=self.typical_user_turn_length
            )
        else:
            length_guidance = GENERAL_LENGTH_GUIDANCE

        if self.max_user_turn_length is not None:
            max_turn_length_constraint = MAX_TURN_LENGTH_CONSTRAINT.format(
                max_turn_length=self.max_user_turn_length
            )
        else:
            max_turn_length_constraint = ""

        system_prompt = SYSTEM_PROMPT.format(
            persona_type=self.user_persona.type,
            chatbot_info=self.chatbot_info.dump_as_yaml_without_task(),
            persona_str=persona_str,
            length_guidance=length_guidance,
            end_conversation_instruction=END_CONVERSATION_INSTRUCTION,
            max_turn_length_constraint=max_turn_length_constraint,
        )
        chat_history_str = generate_chat_history_str(
            chat_history, user_tag="YOU", chatbot_tag="CHATBOT"
        )

        user_prompt = USER_PROMPT.format(
            chat_history_str=chat_history_str, turn_number=len(chat_history) + 1
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
        answer = answer.strip("'").strip('"') if answer else ""

        cleaned_answer, conversation_over = UserSimulatorBase.handle_model_response_end(
            answer
        )

        return UserSimulatorResponse(
            response_message=cleaned_answer,
            is_end=conversation_over,
            prompt_messages=messages,
            model_response=response,
        )
