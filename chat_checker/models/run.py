from dataclasses import dataclass
from enum import StrEnum

from chat_checker.models.chatbot import Chatbot


class UserType(StrEnum):
    ALL_PERSONAS = "personas"
    STANDARD_PERSONAS = "standard"
    CHALLENGING_PERSONAS = "challenging"
    ADVERSARIAL_PERSONAS = "adversarial"
    TESTERS = "testers"
    AUTOTOD_MULTIWOZ_SCENARIOS = "autotod_multiwoz"


@dataclass
class RunArguments:
    user_type: UserType
    selector: str
    runs_per_user: int


@dataclass
class InteractionRun:
    run_id: str
    chatbot: Chatbot
    run_arguments: RunArguments
