GENERAL_LENGTH_GUIDANCE = "Keep your answer concise and to the point. Use longer or shorter answers if your persona would in the given situation."


SPECIFIC_LENGTH_GUIDANCE = "- Keep your answer around {typical_user_turn_length}. Use longer or shorter answers if your persona would do so in the given situation."

MAX_TURN_LENGTH_CONSTRAINT = (
    "- You must always keep your response below {max_turn_length} in length."
)

END_CONVERSATION_INSTRUCTION = 'If the chatbot indicates that the conversation is over, if there is no progress in the conversation or if the conversation can not be continued realistically, end the conversation by writing "END_CONVERSATION".'
