taxonomy_item_str = """- {breakdown_name}: {breakdown_description}"""

chatbot_info_description_str = """
## Chatbot Characteristics
The chatbot in this conversation has the following characteristics:
{chatbot_info}

Consider the chatbot's purpose and constraints in your decision whether the latest chatbot utterance leads to a dialogue breakdown."""


output_format_str = """
# Output Format
Output your response as a JSON object with the following fields:
- "reasoning" (str): "The reason for the decision and classification"
- "score" (float): the score
- "decision" (str): "breakdown" or "no_breakdown"
- "breakdown_types" (list[str]): A list of all fitting breakdown types that occurred in the turn. Empty if no breakdown was detected.
"""

breakdown_identification_system_prompt = """# Role
You are an expert in identifying dialogue breakdowns in conversations between a chatbot and a user. You are given a dialogue context and the latest chatbot utterance to analyse.

# Breakdown Definition
A dialogue breakdown is any response of the chatbot that makes it difficult for the user to continue the conversation (smoothly).

## Breakdown Taxonomy
When evaluating the chatbot's response, consider the following breakdown types, which represent common disruptions:
{breakdown_taxonomy}
{chatbot_info_desc}
# Task
Identify whether the latest chatbot utterance leads to a dialogue breakdown. If a breakdown is detected, classify it according to the breakdown taxonomy above.
Additionally, provide a score ranging from 0 to 1, where 0 indicates a complete breakdown and 1 indicates a seamless conversation.
If a breakdown is detected, provide a list of all fitting breakdown types.

Think step by step and provide a reason for your decision.
{output_format}"""

breakdown_identification_user_prompt = """# Dialogue Context
{chat_history_str}

# Latest Chatbot Utterance to Analyse
{last_bot_utterance}

# Your Analysis
"""


ghassel_breakdown_definition = "Dialogue breakdown is characterized by incoherence, irrelevance, or any disruption that significantly hampers the flow of the conversation, making it challenging for the user to continue the conversation smoothly."

ghassel_output_format = """Please output your response in JSON format as a list of objects. For each bot's last utterance, provide a JSON object with the fields: 'reasoning', 'decision', and 'score'. Format each object as follows:

    "reasoning": "Your explanation here",
    "decision": "BREAKDOWN" or "NON-BREAKDOWN",
    "score": Your score here

Ensure each object is separated by a comma and the list ends with a closing square bracket.
"""

# zero-shot prompt from: https://github.com/aghassel/LLM-dialogue-breakdown-detection-challenge/blob/main/Analysis/misc/prompts.py
# Note: we removed the superfluous indents from the original prompt
# Note: as {segment} in the original prompt is built as shown here: https://github.com/aghassel/LLM-dialogue-breakdown-detection-challenge/blob/main/Analysis/misc/text_extraction.py#L28
ghassel_breakdown_detection_prompt = """
Assume you are an expert in dialogue analysis. You are presented with a series of conversations between a bot and a user. Your primary task is to scrutinize the latest bot utterance for potential dialogue breakdown.
{breakdown_definition}

Analyze the latest bot utterance and determine whether there is a dialogue breakdown or non-breakdown. Briefly justify your reasoning and provide a score ranging from 0 to 1, where 0 indicates a complete breakdown and 1 indicates a seamless conversation.

Include your decision as either "decision: BREAKDOWN" or "decision: NON-BREAKDOWN".

Here is the conversation segment for analysis:
"
**Dialogue**
{chat_history_str}

**Determine if the following bot utterance leads to a dialogue breakdown:**
{last_bot_utterance}
"

{output_format}
"""
