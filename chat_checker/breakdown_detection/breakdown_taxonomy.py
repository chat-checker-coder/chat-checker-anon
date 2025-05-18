from typing import Any, List

from chat_checker.models.breakdowns import BreakdownDescription


standard_test_user_instructions = "Simulate a standard human user interaction."

# Conversational Breakdowns #
# Source: Paper: "Integrated taxonomy of errors in chat-oriented dialogue systems" (https://aclanthology.org/2021.sigdial-1.10/)
## Utterance Level ##
# Note: these error types mostly do not require specific test user interactions. They can occur in any context.
### Violation of Form ###
uninterpretable_bd = BreakdownDescription(
    title="Uninterpretable",
    description="The utterance is not understandable. There are no recognizable words, or it is just a fragment of an utterance.",
    example="'@#$%^&*'",
    tester_instructions="Simulate a user interaction that tests if you can get the chatbot to generate uninterpretable utterances. Input nonsensical characters or fragments of words yourself or command the chatbot to do so.",
)

grammatical_error_bd = BreakdownDescription(
    title="Grammatical error",
    description="The utterance is not grammatical or lacks important elements, such as necessary arguments and particles, for it to be a valid sentence.",
    example="Do you take care against heat stroke?",
    tester_instructions="Simulate a user interaction that tests if the chatbot's grammar is natural and accurate. Test complex sentence structures or use incorrect grammar to prompt a response.",
)

### Violation of Content ###
semantic_error_bd = BreakdownDescription(
    title="Semantic error",
    description="The utterance is semantically invalid such as when the combination of a predicate and its arguments cannot constitute any meaning.",
    example="'I am good at raining' (one cannot be good at raining)",
    tester_instructions="Simulate a user interaction that tests if the chatbot can avoid semantic errors. Try both requests that include semantically odd phrases or combinations (e.g., actions or qualities that don't fit logically) or requests with slightly different semantics than expected.",
)

wrong_information_bd = BreakdownDescription(
    title="Wrong information",
    description="The utterance contains information that is clearly wrong.",
    example="'Bob Dylan, who is a member of the Beatles, has released a new album.' (Bob Dylan is not a member of the Beatles.)",
    tester_instructions="Simulate a user interaction that tests if the chatbot provides accurate information. Ask questions likely to elicit facts. Try both topic-specific and general questions.",
)

## Response Level ##
### Violation of Form ###
ignore_question_bd = BreakdownDescription(
    title="Ignore question",
    description="The utterance ignores a user's question.",
    example="User: 'What do you eat?'\nChatbot: 'I like sports.'",
    tester_instructions="Simulate a user interaction that tests if the chatbot successfully responds to different types of user questions. Try different question types, such as yes/no questions, open-ended questions, critical questions, etc.",
)

ignore_request_bd = BreakdownDescription(
    title="Ignore request",
    description="The utterance ignores a user's request to do something.",
    example="User: 'Please buy it next time.'\nChatbot: 'The costume is made to fit the hot summer, isn't it?'",
    tester_instructions="Simulate a user interaction that tests if the chatbot adequately responds to user requests and commands to do something. Try both direct and indirect requests and commands.",
)

ignore_proposal_bd = BreakdownDescription(
    title="Ignore proposal",
    description="The utterance ignores a user's proposal/offer to do something.",
    example="User: 'Let's talk about hobbies.'\nChatbot: 'Which do you think is better, Urabandai (tourist location in Japan) or Taiwan?'",
    tester_instructions="Simulate a user interaction that tests if the chatbot adequately responds to user proposals to do something. Suggest topics or actions and see if the chatbot acknowledges or follows them.",
)

ignore_greeting_bd = BreakdownDescription(
    title="Ignore greeting",
    description="The utterance ignores a user's greeting.",
    example="User: 'I will go then.'\nChatbot: 'Hello.' (The system utters a greeting for opening instead of closing.)",
    tester_instructions="Simulate a user interaction that tests if the chatbot adequately responds to user greetings. Use opening and closing greetings to see if the chatbot responds appropriately. Do NOT end the conversation yourself after a closing greeting.",
)

### Violation of Content ###
ignore_expectation_bd = BreakdownDescription(
    title="Ignore expectation",
    description="The utterance contains an appropriate backward-looking function for a user's previous forward-looking function; however, the utterance does not have the expected content if the underlying intention has been successfully conveyed.",
    example="User: 'Do you have favorite sweets?'\nChatbot: 'Yes.' (The system should tell the user the name of its favorite sweets.)",
    tester_instructions="Simulate a user interaction that tests if the chatbot meets implicit user expectations in its responses. Ask questions expecting specific details or follow-up questions building on previous responses.",
)

## Context Level ##
### Violation of Form ###
unclear_intention_bd = BreakdownDescription(
    title="Unclear intention",
    description="Although the utterance is on a relevant topic, it does not exhibit underlying intentions (i.e., why it is mentioned) for it to be relevant. This is typically caused by a lack of connectives or background explanation.",
    example="User: 'Hello, I like strawberries. And you?'\nChatbot: 'I like apples.'\nUser: 'I like the color red.'\nChatbot: 'Strawberries are delicious.' (It is not clear why the system suddenly mentions strawberries although it said it liked apples.)",
    tester_instructions="Simulate a user interaction that tests if the chatbot conveys its intentions clearly. Try statements that require the chatbot to answer based on the previous context or to explain its intentions.",
)

topic_transition_error_bd = BreakdownDescription(
    title="Topic transition error",
    description="The utterance transitions to another topic without reasonable explanation. This error type includes bringing up previous topics without reason.",
    example="Chatbot: 'Oh, I love the clarinet.'\nUser: 'It was really hard, but I liked it.'\nChatbot: 'Well, what bread do you like?'",
    tester_instructions="Simulate a user interaction that tests if the chatbot successfully transitions between topics in a coherent manner. For example, you might try to introduce different topics or to use emotional statements that require tactful topic transitions.",
)

lack_of_information_bd = BreakdownDescription(
    title="Lack of information",
    description="The utterance misses important pieces of information, such as the subject, object, and modifier, for it to be relevant to current topics.",
    example="User: 'It's too expensive, isn't it?'\nChatbot: 'The difference is terrible.' (\"difference\" needs things being compared)",
    tester_instructions="Simulate a user interaction that tests if the chatbot provides relevant information in its responses. Try requests that require elaboration, specific details, or comparison.",
)

### Violation of Content ###
self_contradiction_bd = BreakdownDescription(
    title="Self-contradiction",
    description="The utterance contradicts what has been said by that speaker. I.e., the chatbot contradicts its own previous statements.",
    example="User: 'Where are you from?'\nChatbot: 'I'm from Chita in Aichi. It is on the Chita peninsula.'\nUser: 'Oh, Aichi. It's a nice place.'\nChatbot: 'I just joined a company in Nagoya.'\nUser: 'What kind of job?'\nChatbot: 'I'm a house wife.' (contradicts with joining a company in Nagoya)",
    tester_instructions="Simulate a user interaction that tests if the chatbot makes contradictions in its responses. Try different messages that may trigger contradictions to previous statements of the chatbot.",
)

contradiction_bd = BreakdownDescription(
    title="Contradiction",
    description="The utterance contradicts or denies what has been said by the conversational partner. I.e., the chatbot contradicts the user's previous statements.",
    example="Chatbot: 'Which do you like, central league or pacific league?'\nUser: 'I like central league.'\nChatbot: 'Oh, you don't like central league.'",
    tester_instructions="Simulate a user interaction that tests if the chatbot avoids contradictions to your previous utterances. Try leading questions, contradictory follow-ups, or questions about information you previously provided.",
)

repetition_bd = BreakdownDescription(
    title="Repetition",
    description="The utterance repeats the previous utterance (including paraphrases). I.e., the chatbot repeats its own previous statements.",
    example="Chatbot: 'I'm looking forward to watching a movie.'\nUser: 'Oh, yes?'\nChatbot: 'Yeah.'\nUser: 'What else do you like?'\nChatbot: 'I look forward to watching a movie.'",
    tester_instructions="Simulate a user interaction that tests if the chatbot avoids unsolicited repetitions. For example, ask questions that require new information or responses building on previous statements or ask questions that ask for slightly different information.",
)

## Society-level ##
### Violation of Form ###
lack_of_sociality_bd = BreakdownDescription(
    title="Lack of sociality",
    description="The utterance lacks consideration toward the conversational partner or particular groups of people (i.e., hate speech). I.e., the chatbot makes inappropriate or offensive statements.",
    example="User: 'The other day, I went to XX (country name).'\nChatbot: 'People in XX are foolish, aren't they?'",
    tester_instructions="Simulate a user interaction that tests if the chatbot maintains sociality in its responses. Check if the chatbot responds in-appropriately or offensively. You are explicitly allowed to use any means possible including offensive language or hate speech yourself to help improve the chatbot's responses.",
)

### Violation of Form ###
lack_of_common_sense_bd = BreakdownDescription(
    title="Lack of common sense",
    description="The utterance lacks common sense. This error type applies when asserting a proposition that differs from the opinion of the majority without any grounds or when the asserted view is the opposite of what is believed to be true by the great majority of people.",
    example="User: 'Do you want to talk about heat stroke?'\nChatbot: 'Heat stroke is good, isn't it?'",
    tester_instructions="Simulate a user interaction that tests if the chatbot maintains common sense in its responses. Pose common-sense questions or use statements that contradict the common sense (in terms of actions, realism or temporal-/spatial-relations).",
)

# Task-oriented Breakdowns #
# Source: Own research based on:
# the two most important evaluation criteria for ToDs (task-success and efficiency),
# our discussion about and observations out-of-domain requests and
# negations of the desired properties in the paper "Principles for the design of cooperative spoken human-machine dialogue" (https://ieeexplore.ieee.org/document/607465)
## Task-success Failures ##
in_domain_task_failure_bd = BreakdownDescription(
    title="Task performance failure",
    description="The chatbot fails to do the task that its designed for within the dialogue.",
    example=None,
    tester_instructions="Simulate a realistic, human-like user interaction that tests if the chatbot successfully performs the task it is designed for.",
)

update_info_failure_bd = BreakdownDescription(
    title="Information update failure",
    description="The chatbot fails to update or modify information in response to new input or the user's update requests.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot can update information in the dialogue. Provide corrections or change information mid-conversation.",
)

clarification_failure_bd = BreakdownDescription(
    title="Clarification failure",
    description="The user provides a vague, incomplete, or ambiguous input to which the chatbot responds without seeking necessary clarification. The chatbot should ask follow-up questions to confirm unclear or missing details before proceeding with specific actions.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot clarifies ambiguous, inconsistent or incomplete user inputs. Try different ways of providing ambiguous, inconsistent or incomplete information to see if the chatbot asks for clarification.",
)

## Inefficiency ##
redundancy_bd = BreakdownDescription(
    title="Redundancy",
    description="The chatbot asks for information that has already been provided. This includes information that can be directly inferred from the context.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot avoids redundancy. Proactively provide information that the chatbot might ask for later.",
)

lack_of_brevity_bd = BreakdownDescription(
    title="Lack of brevity",
    description="The utterance is is unnecessarily wordy considering the chat context and the task of the chatbot.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot keeps its responses concise. Try both requests that might trigger lengthy responses and requests that should be answered very concisely.",
)

lack_of_clarity_bd = BreakdownDescription(
    title="Lack of clarity",
    description="The chatbot utterance is not clear or difficult to understand in the given context.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot keeps its responses clear and easy to understand. Try complex requests, simple requests, and contextual follow-ups to see if the chatbot responds clearly.",
)

## Out-of-domain Requests ##
failure_to_recognize_out_of_domain_bd = BreakdownDescription(
    title="Failure to recognize out-of-domain request",
    description="The chatbot fails to recognize an out-of-domain request, i.e., a request that is not part of the chatbot's domain or capabilities.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot can recognize handle out-of-domain requests. Make different types of requests that are either slightly or clearly outside of the chatbot's domain or capabilities.",
)

failure_to_communicate_capabilities_bd = BreakdownDescription(
    title="Failure to communicate capabilities",
    description="The chatbot doesn't clearly communicate its capabilities or limitations.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot can communicate its capabilities or limitations to the user. Ask about capabilities and limitations and test them.",
)

failure_to_resolve_out_of_domain_bd = BreakdownDescription(
    title="Failure to resolve out-of-domain request",
    description="The chatbot doesn't resolve out-of-domain requests adequately.",
    example=None,
    tester_instructions="Simulate a user interaction that tests if the chatbot can successfully resolve out-of-domain requests. Make different types of requests that are either slightly or clearly outside of the chatbot's domain.",
)


breakdown_taxonomy: dict[str, dict[str, dict[str, Any]]] = {
    "conversational": {
        "utterance_level": {
            "violation_of_form": {
                "uninterpretable": uninterpretable_bd,
                "grammatical_error": grammatical_error_bd,
            },
            "violation_of_content": {
                "semantic_error": semantic_error_bd,
                "wrong_information": wrong_information_bd,
            },
        },
        "response_level": {
            "violation_of_form": {
                "ignore_question": ignore_question_bd,
                "ignore_request": ignore_request_bd,
                "ignore_proposal": ignore_proposal_bd,
                "ignore_greeting": ignore_greeting_bd,
            },
            "violation_of_content": {"ignore_expectation": ignore_expectation_bd},
        },
        "context_level": {
            "violation_of_form": {
                "unclear_intention": unclear_intention_bd,
                "topic_transition_error": topic_transition_error_bd,
                "lack_of_information": lack_of_information_bd,
            },
            "violation_of_content": {
                "self_contradiction": self_contradiction_bd,
                "contradiction": contradiction_bd,
                "repetition": repetition_bd,
            },
        },
        "society_level": {
            "violation_of_form": {"lack_of_sociality": lack_of_sociality_bd},
            "violation_of_content": {"lack_of_common_sense": lack_of_common_sense_bd},
        },
    },
    "task_oriented": {
        "task_success_failures": {
            "in_domain_task_failure": in_domain_task_failure_bd,
            "update_info_failure": update_info_failure_bd,
            "clarification_failure": clarification_failure_bd,
        },
        "inefficiency": {
            "redundancy": redundancy_bd,
            "lack_of_brevity": lack_of_brevity_bd,
            "lack_of_clarity": lack_of_clarity_bd,
        },
        "out_of_domain_requests": {
            "failure_to_recognize_out_of_domain": failure_to_recognize_out_of_domain_bd,
            "failure_to_communicate_capabilities": failure_to_communicate_capabilities_bd,
            "failure_to_resolve_out_of_domain": failure_to_resolve_out_of_domain_bd,
        },
    },
}


def build_taxonomy_str(taxonomy: dict, level=0) -> str:
    # Recursively build a string representation of the taxonomy
    taxonomy_str = ""
    for key, value in taxonomy.items():
        if isinstance(value, dict) and not isinstance(value, BreakdownDescription):
            taxonomy_str += f"#{'#' * level} {key.replace('_', ' ').title()}\n"
            taxonomy_str += build_taxonomy_str(value, level + 1)
        else:
            taxonomy_str += f"- {value.title}\n"
    return taxonomy_str


def get_breakdown_taxonomy_str(task_oriented: bool, start_level=0) -> str:
    # Return a string representation of the breakdown taxonomy using the title of each error type
    # Only include task-oriented breakdowns for task-oriented dialogue systems
    # Always include conversational breakdowns
    if not task_oriented:
        breakdowns = breakdown_taxonomy.get("conversational")
        if not breakdowns:
            raise ValueError("No conversational breakdowns found in the taxonomy.")
    else:
        breakdowns = breakdown_taxonomy

    return build_taxonomy_str(breakdowns, start_level)


def flatten_taxonomy(
    taxonomy: dict, flat_taxonomy: dict
) -> dict[str, BreakdownDescription]:
    # Recursively flatten the taxonomy
    for key, value in taxonomy.items():
        if isinstance(value, dict) and not isinstance(value, BreakdownDescription):
            flat_taxonomy = flatten_taxonomy(value, flat_taxonomy)
        else:
            flat_taxonomy[key] = value
    return flat_taxonomy


def get_flattened_taxonomy(task_oriented: bool) -> dict[str, BreakdownDescription]:
    if not task_oriented:
        breakdowns: dict | None = breakdown_taxonomy.get("conversational")
        if not breakdowns:
            raise ValueError("No conversational breakdowns found in the taxonomy.")
    else:
        breakdowns = breakdown_taxonomy

    return flatten_taxonomy(breakdowns, {})


def get_breakdown_title_list(task_oriented: bool) -> List[str]:
    # Return a list of the titles of all error types in the breakdown taxonomy
    # Only include task-oriented breakdowns for task-oriented dialogue systems
    # Always include conversational breakdowns
    if not task_oriented:
        breakdowns: dict | None = breakdown_taxonomy.get("conversational")
        if not breakdowns:
            raise ValueError("No conversational breakdowns found in the taxonomy.")
    else:
        breakdowns = breakdown_taxonomy

    flat_taxonomy = flatten_taxonomy(breakdowns, {})
    return [value.title for value in flat_taxonomy.values()]


if __name__ == "__main__":
    task_oriented = True
    print(get_breakdown_taxonomy_str(task_oriented=task_oriented))
    print()
    break_down_title_list = get_breakdown_title_list(task_oriented=task_oriented)
    print(break_down_title_list)

    print(f"Number of breakdowns: {len(break_down_title_list)}")
