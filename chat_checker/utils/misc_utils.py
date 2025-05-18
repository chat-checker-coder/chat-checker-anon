import os
from pathlib import Path
import re
from typing import List, Optional

import litellm
from pydantic import SecretStr
import numpy as np
from openai.types.chat import ChatCompletionMessageParam
from lexical_diversity import lex_div


from chat_checker.models.dialogue import (
    Dialogue,
    DialogueTurn,
    FinishReason,
    SpeakerRole,
)
from chat_checker.models.llm import UsageCost  # type: ignore

BASE_DIR = Path(__file__).parent
OPENAI_API_KEY_NAME = "CHAT_CHECKER_OPENAI_API_KEY"
GEMINI_API_KEY_NAME = "CHAT_CHECKER_GEMINI_API_KEY"
ANTHROPIC_API_KEY_NAME = "CHAT_CHECKER_ANTHROPIC_API_KEY"


def safe_load_api_key(api_key: str) -> Optional[SecretStr]:
    key = os.getenv(api_key)
    if not key:
        return None
    return SecretStr(key)


OPENAI_API_KEY = safe_load_api_key(OPENAI_API_KEY_NAME) or SecretStr(
    "no_api_key_provided"
)

GEMINI_API_KEY = safe_load_api_key(GEMINI_API_KEY_NAME) or SecretStr(
    "no_api_key_provided"
)

ANTHROPIC_API_KEY = safe_load_api_key(ANTHROPIC_API_KEY_NAME) or SecretStr(
    "no_api_key_provided"
)


def get_matching_api_key(model_name: str) -> SecretStr:
    if model_name.startswith("claude/"):
        return ANTHROPIC_API_KEY
    elif model_name.startswith("gemini/"):
        return GEMINI_API_KEY
    elif (
        model_name in litellm.open_ai_chat_completion_models
        or model_name in litellm.open_ai_text_completion_models
    ):
        return OPENAI_API_KEY
    else:
        raise ValueError(f"Model {model_name} is not supported yet.")


def verify_environment(is_cli=False) -> bool:
    if not safe_load_api_key(OPENAI_API_KEY_NAME):
        if is_cli:
            print(
                f"Please set the OpenAI API key in the environment variable {OPENAI_API_KEY_NAME}."
            )
        else:
            raise ValueError(
                f"Please set the OpenAI API key in the environment variable {OPENAI_API_KEY_NAME}."
            )
        return False
    return True


def five_num_summary(data):
    # Filter out nan values
    data = [x for x in data if not np.isnan(x)]
    return {
        "min": float(np.min(data)) if data else None,
        "q1": float(np.percentile(data, 25)) if data else None,
        "median": float(np.median(data)) if data else None,
        "q3": float(np.percentile(data, 75)) if data else None,
        "max": float(np.max(data)) if data else None,
    }


def compute_chat_statistics(chat_history: List[DialogueTurn]) -> dict:
    num_turns = len(chat_history)
    user_turns = [
        message for message in chat_history if message.role == SpeakerRole.USER
    ]
    chatbot_turns = [
        message
        for message in chat_history
        if message.role == SpeakerRole.DIALOGUE_SYSTEM
    ]
    num_user_turns = len(user_turns)
    num_chatbot_turns = len(chatbot_turns)
    if num_user_turns == 0:
        avg_user_turn_length = 0.0
        five_num_summary_user_turns = None
    else:
        user_turn_word_lengths = [
            len(message.content.split()) for message in user_turns
        ]
        avg_user_turn_length = sum(user_turn_word_lengths) / num_user_turns
        five_num_summary_user_turns = five_num_summary(user_turn_word_lengths)
    if num_chatbot_turns == 0:
        avg_chatbot_turn_length = 0.0
        five_num_summary_chatbot_turns = None
    else:
        chatbot_turn_word_lengths = [
            len(message.content.split()) for message in chatbot_turns
        ]
        avg_chatbot_turn_length = sum(chatbot_turn_word_lengths) / num_chatbot_turns
        five_num_summary_chatbot_turns = five_num_summary(chatbot_turn_word_lengths)
    return {
        "num_turns": num_turns,
        "num_user_turns": num_user_turns,
        "num_chatbot_turns": num_chatbot_turns,
        "avg_user_turn_length": avg_user_turn_length,
        "five_num_summary_user_turn_lengths": five_num_summary_user_turns,
        "avg_chatbot_turn_length": avg_chatbot_turn_length,
        "five_num_summary_chatbot_turn_lengths": five_num_summary_chatbot_turns,
    }


def compute_run_statistics(dialogues: List[Dialogue]) -> dict:
    num_dialogues = len(dialogues)
    dialogues_with_errors = [
        dialogue for dialogue in dialogues if dialogue.error is not None
    ]
    dialogues_with_chatbot_errors = [
        dialogue
        for dialogue in dialogues_with_errors
        if dialogue.finish_reason == FinishReason.CHATBOT_ERROR
    ]
    dialogues_with_simulator_errors = [
        dialogue
        for dialogue in dialogues_with_errors
        if dialogue.finish_reason == FinishReason.USER_SIMULATOR_ERROR
    ]

    if num_dialogues != 0:
        dialogue_chat_statistics = [
            dialogue.chat_statistics
            for dialogue in dialogues
            if dialogue.chat_statistics
        ]
        num_user_turns = sum(
            [
                chat_statistics["num_user_turns"]
                for chat_statistics in dialogue_chat_statistics
            ]
        )
        avg_user_turns_per_dialogue = num_user_turns / num_dialogues
        num_chatbot_turns = sum(
            [
                chat_statistics["num_chatbot_turns"]
                for chat_statistics in dialogue_chat_statistics
            ]
        )
        avg_chatbot_turns_per_dialogue = num_chatbot_turns / num_dialogues
        avg_avg_user_turn_length = (
            sum(
                [
                    chat_statistics["avg_user_turn_length"]
                    for chat_statistics in dialogue_chat_statistics
                ]
            )
            / num_dialogues
        )
        avg_avg_chatbot_turn_length = (
            sum(
                [
                    chat_statistics["avg_chatbot_turn_length"]
                    for chat_statistics in dialogue_chat_statistics
                ]
            )
            / num_dialogues
        )
        five_num_summary_user_turns = five_num_summary(
            [
                chat_statistics["num_user_turns"]
                for chat_statistics in dialogue_chat_statistics
            ]
        )
        five_num_summary_chatbot_turns = five_num_summary(
            [
                chat_statistics["num_chatbot_turns"]
                for chat_statistics in dialogue_chat_statistics
            ]
        )

        all_user_turns = [
            turn
            for dialogue in dialogues
            for turn in dialogue.chat_history
            if turn.role == SpeakerRole.USER
        ]

        all_chatbot_turns = [
            turn
            for dialogue in dialogues
            for turn in dialogue.chat_history
            if turn.role == SpeakerRole.DIALOGUE_SYSTEM
        ]

        chatbot_turn_lengths = [len(turn.content.split()) for turn in all_chatbot_turns]
        user_turn_lengths = [len(turn.content.split()) for turn in all_user_turns]
        avg_user_turn_length = sum(user_turn_lengths) / len(all_user_turns)
        avg_chatbot_turn_length = sum(chatbot_turn_lengths) / len(all_chatbot_turns)

        five_num_summary_user_turn_length = five_num_summary(user_turn_lengths)
        five_num_summary_chatbot_turn_length = five_num_summary(chatbot_turn_lengths)

        all_user_turns = [
            turn
            for dialogue in dialogues
            for turn in dialogue.chat_history
            if turn.role == SpeakerRole.USER
        ]
        all_chatbot_turns = [
            turn
            for dialogue in dialogues
            for turn in dialogue.chat_history
            if turn.role == SpeakerRole.DIALOGUE_SYSTEM
        ]
        all_user_turn_tokens = []
        for turn in all_user_turns:
            tokens = lex_div.tokenize(turn.content)
            all_user_turn_tokens.extend(tokens)
        all_chatbot_turn_tokens = []
        for turn in all_chatbot_turns:
            tokens = lex_div.tokenize(turn.content)
            all_chatbot_turn_tokens.extend(tokens)

        user_turn_mtld = lex_div.mtld(all_user_turn_tokens)
        chatbot_turn_mtld = lex_div.mtld(all_chatbot_turn_tokens)

        run_chat_statistics = {
            "num_dialogues": num_dialogues,
            "num_dialogues_with_errors": len(dialogues_with_errors),
            "num_dialogues_with_chatbot_errors": len(dialogues_with_chatbot_errors),
            "dialogues_with_chatbot_errors": [
                dialogue.dialogue_id for dialogue in dialogues_with_chatbot_errors
            ],
            "num_dialogues_with_simulator_errors": len(dialogues_with_simulator_errors),
            "dialogues_with_simulator_errors": [
                dialogue.dialogue_id for dialogue in dialogues_with_simulator_errors
            ],
            "num_user_turns": num_user_turns,
            "avg_user_turns_per_dialogue": avg_user_turns_per_dialogue,
            "five_num_summary_user_turns": five_num_summary_user_turns,
            "num_chatbot_turns": num_chatbot_turns,
            "avg_chatbot_turns_per_dialogue": avg_chatbot_turns_per_dialogue,
            "five_num_summary_chatbot_turns": five_num_summary_chatbot_turns,
            "avg_avg_user_turn_length": avg_avg_user_turn_length,
            "avg_user_turn_length": avg_user_turn_length,
            "five_num_summary_avg_user_turn_length": five_num_summary_user_turn_length,
            "avg_avg_chatbot_turn_length": avg_avg_chatbot_turn_length,
            "avg_chatbot_turn_length": avg_chatbot_turn_length,
            "five_num_summary_avg_chatbot_turn_length": five_num_summary_chatbot_turn_length,
            "user_turn_mtld": user_turn_mtld,
            "chatbot_turn_mtld": chatbot_turn_mtld,
        }
    else:
        run_chat_statistics = {
            "num_dialogues": 0,
        }

    if num_dialogues != 0:
        # Compute cost statistics
        dialogue_sim_cost_statistics = [
            dialogue.simulation_cost_statistics
            for dialogue in dialogues
            if dialogue.simulation_cost_statistics
        ]
        total_prompt_tokens = sum(
            [
                simulation_cost_statistics["total_prompt_tokens"]
                for simulation_cost_statistics in dialogue_sim_cost_statistics
            ]
        )
        total_completion_tokens = sum(
            [
                simulation_cost_statistics["total_completion_tokens"]
                for simulation_cost_statistics in dialogue_sim_cost_statistics
            ]
        )
        total_tokens = sum(
            [
                simulation_cost_statistics["total_tokens"]
                for simulation_cost_statistics in dialogue_sim_cost_statistics
            ]
        )
        total_cost = sum(
            [
                simulation_cost_statistics["cost"]
                for simulation_cost_statistics in dialogue_sim_cost_statistics
            ]
        )
        avg_prompt_tokens = total_prompt_tokens / num_dialogues
        avg_completion_tokens = total_completion_tokens / num_dialogues
        avg_total_tokens = total_tokens / num_dialogues
        avg_cost = total_cost / num_dialogues
        five_num_summary_prompt_tokens = five_num_summary(
            [
                simulation_cost_statistics["total_prompt_tokens"]
                for simulation_cost_statistics in dialogue_sim_cost_statistics
            ]
        )
        five_num_summary_total_cost = five_num_summary(
            [
                simulation_cost_statistics["cost"]
                for simulation_cost_statistics in dialogue_sim_cost_statistics
            ]
        )

        run_cost_statistics = {
            "total_prompt_tokens": total_prompt_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "five_num_summary_prompt_tokens": five_num_summary_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "total_tokens": total_tokens,
            "avg_total_tokens": avg_total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_dialogue": avg_cost,
            "five_num_summary_cost_per_dialogue": five_num_summary_total_cost,
        }
    else:
        run_cost_statistics = {
            "total_prompt_tokens": 0,
        }

    return {
        "run_chat_statistics": run_chat_statistics,
        "run_cost_statistics": run_cost_statistics,
    }


def fill_in_persona_type(user_persona: dict):
    persona_id = user_persona.get("id", None)
    persona_type = user_persona.get("persona_type", None)
    if persona_type is None:
        if persona_id is None:
            raise ValueError("User persona does not have a type or ID.")
        re_match = re.match(r"generated_(.+)_persona_\d+", persona_id)
        if re_match:
            persona_type = re_match.group(1)
            user_persona["persona_type"] = persona_type
        else:
            print(
                f"Warning: Could not extract persona type from persona ID {persona_id}. Setting persona type to 'standard'."
            )
            user_persona["persona_type"] = "standard"


def compute_analysis_cost_statistics(
    dialogues: List[Dialogue], total_analysis_usage: UsageCost
) -> dict:
    avg_prompt_tokens = total_analysis_usage.prompt_tokens / len(dialogues)
    avg_completion_tokens = total_analysis_usage.completion_tokens / len(dialogues)
    avg_total_tokens = total_analysis_usage.total_tokens / len(dialogues)
    avg_cost = total_analysis_usage.cost / len(dialogues)
    cost_stats = {
        "prompt_tokens": total_analysis_usage.prompt_tokens,
        "completion_tokens": total_analysis_usage.completion_tokens,
        "total_tokens": total_analysis_usage.total_tokens,
        "cost": total_analysis_usage.cost,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_cost": avg_cost,
    }
    return cost_stats


def write_prompt_to_txt_file(
    prompt: list[ChatCompletionMessageParam], file: Path
) -> None:
    prompt_str = ""
    for message in prompt:
        if message["role"] == "system":
            prompt_str += f"System prompt:\n{message['content']}\n"
        elif message["role"] == "user":
            prompt_str += f"User prompt:\n{message['content']}\n"
        elif message["role"] == "assistant":
            prompt_str += f"Assistant response:\n{message['content']}\n"
    with open(file, "w", encoding="utf-8") as f:
        f.write(prompt_str)
