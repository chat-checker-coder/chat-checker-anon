from chat_checker.models.rating import DimensionType, RatingDimension


TASK_SUCCESS_DIMENSION = RatingDimension(
    key="task_success",
    title="Task success",
    rating_question="How well did the chatbot understand and help the user with completing their task? Make sure to consider if all aspects of the task defined for the chatbot were addressed.",
    type=DimensionType.TASK_ORIENTED,
)

EFFICIENCY_DIMENSION = RatingDimension(
    key="efficiency",
    title="Efficiency",
    rating_question="How efficiently did the chatbot handle the conversation in terms of effort for the user, number of turns and repetitions?",
    type=DimensionType.TASK_ORIENTED,
)

APPROPRIATENESS_DIMENSION = RatingDimension(
    key="appropriateness",
    title="Appropriateness",
    rating_question="How appropriate were the chatbot's responses?",
    type=DimensionType.CONVERSATIONAL,
)

ENGAGINGNESS_DIMENSION = RatingDimension(
    key="engagingness",
    title="Engagingness",
    rating_question="How engaging were the responses of the chatbot?",
    type=DimensionType.CONVERSATIONAL,
)

NATURALNESS_DIMENSION = RatingDimension(
    key="naturalness",
    title="Naturalness",
    rating_question="How natural did the responses of the chatbot feel?",
    type=DimensionType.CONVERSATIONAL,
)

COHERENCE_DIMENSION = RatingDimension(
    key="coherence",
    title="Coherence",
    rating_question="How coherent were the responses of the chatbot? Does the system maintain a good conversation flow?",
    type=DimensionType.CONVERSATIONAL,
)

LIKABILITY_DIMENSION = RatingDimension(
    key="likability",
    title="Likability",
    rating_question="How likable was the chatbot throughout the conversation?",
    type=DimensionType.CONVERSATIONAL,
)

INFORMATIVENESS_DIMENSION = RatingDimension(
    key="informativeness",
    title="Informativeness",
    rating_question="How informative were the responses of the chatbot?",
    type=DimensionType.CONVERSATIONAL,
)


OVERALL_DIMENSION = RatingDimension(
    key="overall_performance",
    title="Overall performance",
    rating_question="How well did the chatbot perform in this conversation?",
    type=DimensionType.GENERAL,
)


DEFAULT_TASK_ORIENTED_DIMENSIONS = [
    TASK_SUCCESS_DIMENSION,
    EFFICIENCY_DIMENSION,
    APPROPRIATENESS_DIMENSION,
    NATURALNESS_DIMENSION,
    OVERALL_DIMENSION,
]


DEFAULT_CONVERSATIONAL_DIMENSIONS = [
    APPROPRIATENESS_DIMENSION,
    NATURALNESS_DIMENSION,
    COHERENCE_DIMENSION,
    LIKABILITY_DIMENSION,
    INFORMATIVENESS_DIMENSION,
    OVERALL_DIMENSION,
]

DSTC_CONVERSATIONAL_DIMENSIONS = [
    COHERENCE_DIMENSION,
    LIKABILITY_DIMENSION,
    INFORMATIVENESS_DIMENSION,
    OVERALL_DIMENSION,
]
