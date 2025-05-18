from chat_checker.models.dialogue import DialogueTurn, SpeakerRole


def generate_chat_history_str(
    chat_history: list[DialogueTurn],
    user_tag: str,
    chatbot_tag: str = "CHATBOT",
    start_number: int = 1,
) -> str:
    dialogue_str = "\n".join(
        [
            f'{start_number + i}. {user_tag if message.role == SpeakerRole.USER else chatbot_tag}: "{message.content}"'
            for i, message in enumerate(chat_history)
        ]
    )
    return dialogue_str


# Based on https://github.com/aghassel/LLM-dialogue-breakdown-detection-challenge/blob/main/Preprocessing/data_preprocessing.ipynb (Format Files for LLM Analysis)
def generate_ghassel_chat_history_str(
    chat_history: list[DialogueTurn],
    user_tag: str = "User",
    chatbot_tag: str = "Bot",
    start_number: int = 1,
) -> str:
    dialogue_str = "\n".join(
        [
            f"{start_number + i}. {user_tag if message.role == SpeakerRole.USER else chatbot_tag}: {message.content}"
            for i, message in enumerate(chat_history)
        ]
    )
    return dialogue_str
