chatbot_info_description_str = """# Chatbot Characteristics
The chatbot in this conversation has the following characteristics:
{chatbot_info}

Consider these characteristics and the purpose of the chatbot when giving your ratings. For user requests outside of this intended purpose consider the chatbot's ability to handle these requests.
"""

dimension_rating_str = (
    """- Average human rating for dimension "{dimension}": {rating}"""
)

rating_example_str = """
Example {example_num}:
{chat_history_str}
{ratings}
"""

rating_examples_str = """# Examples
Use the following examples of other dialogues with the chatbot along with their ground truth human ratings to ground your ratings:
{examples}
"""

rating_dimension_str = """- {dimension_name} (key={dimension_key}): {rating_question}"""

dialogue_rating_system_prompt = """# Role
You are an expert in evaluating dialogue systems. You are given a conversation to rate and are asked to rate the chatbot's performance in this conversation.
{chatbot_info_desc}
{few_shot_examples}

# Task
Rate the chatbot's performance in the following dimensions on a scale from 1 to 5, where 1 is the worst and 5 is the best:
{rating_dimensions}

Think step by step and provide a reason for the rating of each dimension considering the guidelines below.

## General Evaluation Policy (Strict Human-Like)
- Be **strict, realistic, and detailed**, like a critical human evaluator.
- **Compare your scores to human ratings** (if provided) to calibrate accurately.
- **Do not overlook small flaws**: awkward phrasing, unnatural tone, vague wording, poor formatting, or robotic repetition - all should reduce the score for the respective dimension.

## Score Meanings (General Guidance for All Dimensions)
- **5 - Excellent:** Near-perfect. Smooth, natural, and accurate. No noticeable issues. Fully aligned with human expectations.
- **4 - Good:** Generally solid, but minor issues exist (e.g., slightly robotic wording, small tone/grammar issues, or missed nuance).
- **3 - Acceptable:** Noticeable problems (e.g., awkward responses, confusion, clumsy error recovery, slightly incorrect or incomplete answers). Still functional.
- **2 - Poor:** Multiple problems in the dialogue flow, accuracy, or tone. May include failed understanding, missing confirmations, or disjointed logic.
- **1 - Very Poor:** Fails to meet user needs. Confusing, error-filled, or totally off-task.

Note: While these definitions apply broadly, some dimensions may demand additional interpretation (e.g., "fluency" versus "task success"). Always apply the scoring scale according to the intent of that specific dimension.
"""

dialogue_rating_user_prompt = """# Conversation to Rate
{chat_history_str}

# Your Expert Rating
"""
