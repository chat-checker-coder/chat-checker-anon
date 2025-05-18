from importlib.machinery import SourceFileLoader
import json
from pathlib import Path
from datetime import datetime
import random
from typing import List, Optional
import os
from tqdm import tqdm

import yaml
from litellm.types.utils import ModelResponse

from chat_checker.chatbot_connection.chatbot_client_base import ChatbotClientInterface
from chat_checker.data_management.storage_manager import load_user_personas
from chat_checker.models.breakdowns import (
    BreakdownAnnotation,
    BreakdownDecision,
    BreakdownDescription,
)
from chat_checker.models.chatbot import Chatbot, ChatbotType
from chat_checker.models.dialogue import (
    Dialogue,
    DialogueTurn,
    FinishReason,
    SpeakerRole,
)
from chat_checker.models.llm import UsageCost
from chat_checker.models.run import UserType
from chat_checker.models.user_personas import Persona, PersonaType
from chat_checker.user_simulation.autotod_multiwoz_simulator import (
    AutotodMultiwozSimulator,
)
from chat_checker.user_simulation.user_simulator_base import (
    UserSimulatorBase,
)
from chat_checker.user_simulation.persona_simulator.persona_simulator import (
    PersonaSimulator,
)
from chat_checker.user_simulation.test_user_simulator.test_user_simulator import (
    TestUserSimulator,
)
from chat_checker.utils.llm_utils import compute_total_usage, DEFAULT_LLM
from chat_checker.utils.misc_utils import (
    compute_run_statistics,
    compute_chat_statistics,
)
from chat_checker.breakdown_detection.breakdown_taxonomy import breakdown_taxonomy
from chat_checker.utils.prompt_utils import generate_chat_history_str

BASE_DIR = Path(__file__).parent

DEFAULT_MAX_USER_TURNS = 10


def simulate_dialogues(
    run_id: str,
    user_name: str,
    dialogue_base_dir: Path,
    chatbot_client: ChatbotClientInterface,
    user_simulator: UserSimulatorBase,
    user_simulator_setup_kwargs: dict,
    max_user_turns: int,
    runs_per_user: int = 1,
    save_prompt=False,
) -> list[Dialogue]:
    dialogues: list[Dialogue] = []
    for i in range(runs_per_user):
        print(f"Run {i + 1}/{runs_per_user}")
        start_time = datetime.now()
        chat_history: list[DialogueTurn] = []
        model_responses: list[ModelResponse] = []
        user_simulator.set_up_session(**user_simulator_setup_kwargs)
        first_chatbot_message = chatbot_client.set_up_chat()
        print("--- Conversation Start ---")
        if first_chatbot_message:
            turn_id = 1
            first_turn = DialogueTurn(
                turn_id=turn_id,
                role=SpeakerRole.DIALOGUE_SYSTEM,
                content=first_chatbot_message,
            )
            chat_history.append(first_turn)
            print(f"{turn_id}. Chatbot: {first_chatbot_message}")
        else:
            turn_id = 0
        total_simulation_usage = UsageCost(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.0,
        )
        finish_reason = None
        error = None
        for _ in range(max_user_turns):
            try:
                simulator_response = user_simulator.generate_response(chat_history)
            except Exception as e:
                print(f"Error in getting simulated response: {e}")
                finish_reason = FinishReason.USER_SIMULATOR_ERROR
                error = str(e)
                break
            turn_id = turn_id + 1
            print(f"{turn_id}. USER: {simulator_response.response_message}")
            if save_prompt:
                prompt_dir = f"{dialogue_base_dir}/simulation_prompts/run_{i + 1}"
                os.makedirs(prompt_dir, exist_ok=True)
                prompt_str = ""
                if simulator_response.prompt_messages:
                    for message in simulator_response.prompt_messages:
                        prompt_str += f"{message['role']}: {message['content']}\n\n"
                else:
                    prompt_str = "No prompt messages."
                with open(
                    f"{prompt_dir}/turn_{turn_id}_prompt.txt", "w+", encoding="utf-8"
                ) as f:
                    f.write(prompt_str)
            if simulator_response.model_response is not None:
                model_responses.append(simulator_response.model_response)
                usage = compute_total_usage([simulator_response.model_response])
                total_simulation_usage.prompt_tokens += usage.prompt_tokens
                total_simulation_usage.completion_tokens += usage.completion_tokens
                total_simulation_usage.total_tokens += (
                    usage.prompt_tokens + usage.completion_tokens
                )
                total_simulation_usage.cost += usage.cost
            user_ended_conversation = simulator_response.is_end
            user_message_empty = (
                simulator_response.response_message is None
                or simulator_response.response_message == ""
            )
            if not user_message_empty:
                user_simulator_turn = DialogueTurn(
                    turn_id=turn_id,
                    role=SpeakerRole.USER,
                    content=simulator_response.response_message,
                )
                chat_history.append(user_simulator_turn)
            if user_ended_conversation or user_message_empty:
                finish_reason = FinishReason.USER_ENDED
                break
            try:
                chatbot_response, chatbot_ended_conversation = (
                    chatbot_client.get_response(simulator_response.response_message)
                )
            except Exception as e:
                print(f"Error in getting chatbot response: {e}")
                finish_reason = FinishReason.CHATBOT_ERROR
                error = str(e)
                chatbot_response = finish_reason

            turn_id = turn_id + 1
            if finish_reason == FinishReason.CHATBOT_ERROR:
                error_chatbot_turn = DialogueTurn(
                    turn_id=turn_id,
                    role=SpeakerRole.DIALOGUE_SYSTEM,
                    content=chatbot_response,
                    breakdown_annotation=BreakdownAnnotation(
                        reasoning=f"Received error: {error}",
                        score=0,
                        decision=BreakdownDecision.BREAKDOWN,
                        breakdown_types=["Chatbot Crash"],
                    ),
                )
                chat_history.append(error_chatbot_turn)
                break
            else:
                chatbot_turn = DialogueTurn(
                    turn_id=turn_id,
                    role=SpeakerRole.DIALOGUE_SYSTEM,
                    content=chatbot_response,
                )
                chat_history.append(chatbot_turn)
            print(f"{turn_id}. CHATBOT: {chatbot_response}")
            if chatbot_ended_conversation:
                finish_reason = FinishReason.CHATBOT_ENDED
                break
        end_time = datetime.now()
        print("--- Conversation End ---")
        if finish_reason == FinishReason.CHATBOT_ENDED:
            print("# Conversation ended by chatbot.")
        elif finish_reason == FinishReason.USER_ENDED:
            print("# Conversation ended by user.")
        elif finish_reason == FinishReason.USER_SIMULATOR_ERROR:
            print("# Conversation ended due to an error in the user simulator.")
        elif finish_reason == FinishReason.CHATBOT_ERROR:
            print("# Conversation ended due to an error in the chatbot.")
        else:
            finish_reason = FinishReason.MAX_TURNS_REACHED
            print("# Conversation ended. Maximum number of user messages reached.")
        chatbot_client.tear_down_chat()
        user_simulator.tear_down_session()

        chat_stats = compute_chat_statistics(chat_history)

        chat_stats = {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": (end_time - start_time).total_seconds(),
            **chat_stats,
        }

        cost_stats = {
            "total_prompt_tokens": total_simulation_usage.prompt_tokens,
            "total_completion_tokens": total_simulation_usage.completion_tokens,
            "total_tokens": total_simulation_usage.total_tokens,
            "cost": total_simulation_usage.cost,
        }

        # Write the dialogue to a text file
        dialogue_file_name = f"dialogue_{i + 1}"
        dialogue_id = f"{user_name}_dialogue_{i + 1}"
        os.makedirs(dialogue_base_dir, exist_ok=True)
        dialogue_text_file = f"{dialogue_base_dir}/{dialogue_file_name}.txt"
        dialogue_str = generate_chat_history_str(
            chat_history, user_tag="USER", chatbot_tag="CHATBOT"
        )
        with open(dialogue_text_file, "w", encoding="utf-8") as file:
            file.write("Chat history:\n")
            file.write(dialogue_str)
            file.write(f"\n\n# Finish reason: {finish_reason}")
            file.write("\n\n")

        # Write dialogue to a yaml file
        dialogue_yaml = dialogue_base_dir / f"{dialogue_file_name}.yaml"
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            path=dialogue_yaml,
            user_name=user_name,
            chat_history=chat_history,
            finish_reason=finish_reason,
            error=error,
            chat_statistics=chat_stats,
            simulation_cost_statistics=cost_stats,
        )

        with open(dialogue_yaml, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                dialogue.model_dump(),
                file,
                indent=4,
                sort_keys=False,
                allow_unicode=True,
            )

        dialogues.append(dialogue)
    return dialogues


def run_autotod_multiwoz_simulator(
    run_id: str,
    chatbot: Chatbot,
    chatbot_client: ChatbotClientInterface,
    max_user_turns: int,
    n_dialogues: int,
    seed: Optional[int] = None,
    runs_per_user=1,
) -> list[Dialogue]:
    print(
        f"Simulating {n_dialogues} dialogues with AutoTOD simulator for chatbot {chatbot.id}..."
    )
    run_base_dir = chatbot.base_directory / "runs" / run_id
    # Load MWOZ dialogue ids from mwoz_dialogue_ids.json
    with open(BASE_DIR / "data/mwoz_dialogue_ids.json", "r", encoding="utf-8") as f:
        mwoz_dialogue_ids = json.load(f)

    if seed is not None:
        random.seed(seed)
    sampled_dialogue_ids = random.sample(mwoz_dialogue_ids, n_dialogues)

    all_simulated_dialogues = []
    for i, mwoz_dialogue_id in tqdm(enumerate(sampled_dialogue_ids)):
        truncated_dialogue_id = mwoz_dialogue_id.split(".")[0]
        dialogue_num = i + 1
        # Format the dialogue number to have as many digits as the number of dialogues
        max_dialogue_num_digits = len(str(n_dialogues))
        dialogue_num_str = f"{dialogue_num:0{max_dialogue_num_digits}d}"
        dialogue_base_dir = (
            run_base_dir / f"{dialogue_num_str}_autotod_mwoz_{truncated_dialogue_id}"
        )
        os.makedirs(dialogue_base_dir, exist_ok=True)

        user_info = {
            "run_id": run_id,
            "mwoz_dialogue_id": mwoz_dialogue_id,
        }
        user_info_file = f"{dialogue_base_dir}/user_info.yaml"
        with open(user_info_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(user_info, f, indent=4, sort_keys=False, allow_unicode=True)

        user_simulator = AutotodMultiwozSimulator(
            multiwoz_dialogue_id=mwoz_dialogue_id, seed=seed
        )
        setup_kwargs: dict = {}
        dialogues = simulate_dialogues(
            run_id,
            f"{dialogue_num_str}_autotod_mwoz_{truncated_dialogue_id}",
            dialogue_base_dir,
            chatbot_client,
            user_simulator,
            setup_kwargs,
            max_user_turns=max_user_turns,
            runs_per_user=runs_per_user,
            save_prompt=False,
        )
        all_simulated_dialogues.extend(dialogues)
    return all_simulated_dialogues


def simulate_testers(
    run_id: str,
    chatbot: Chatbot,
    chatbot_client: ChatbotClientInterface,
    breakdowns_to_test: str,
    max_user_turns: int,
    typical_user_turn_length: Optional[str] = None,
    max_user_turn_length: Optional[str] = None,
    runs_per_breakdown=1,
    save_prompt=False,
    user_simulator_llm: str = DEFAULT_LLM,
    seed: Optional[int] = None,
) -> List[Dialogue]:
    run_base_dir = chatbot.base_directory / "runs" / run_id
    keys = breakdowns_to_test.split(".") if breakdowns_to_test != "" else []
    is_task_oriented = chatbot.info.type == ChatbotType.TASK_ORIENTED
    breakdowns = (
        breakdown_taxonomy
        if is_task_oriented
        else breakdown_taxonomy.get("conversational", {})
    )
    for key in keys:
        taxonomy_item = breakdowns.get(key, None)
        # Handle the case where we are testing a single breakdown
        if taxonomy_item is None:
            error_str = f"Breakdown {key} not found in the breakdown taxonomy."
            raise ValueError(error_str)
        elif taxonomy_item.get("title") is not None:
            breakdowns = {key: taxonomy_item}
        else:
            # go deeper in the taxonomy
            breakdowns = taxonomy_item

    all_simulated_dialogues = []
    for key, bd in breakdowns.items():
        if type(bd) is dict:
            # go deeper in the taxonomy
            next_breakdowns_to_test = (
                breakdowns_to_test + "." + key if breakdowns_to_test != "" else key
            )
            dialogues = simulate_testers(
                run_id,
                chatbot,
                chatbot_client,
                next_breakdowns_to_test,
                max_user_turns,
                # TODO: consider including typical_user_turn_length in the recursive call
                max_user_turn_length=max_user_turn_length,
                runs_per_breakdown=runs_per_breakdown,
                save_prompt=save_prompt,
                user_simulator_llm=user_simulator_llm,
                seed=seed,
            )
            all_simulated_dialogues.extend(dialogues)
        elif type(bd) is BreakdownDescription:
            print(f"Simulating testers for breakdown: {bd.title}")
            full_breakdown_key = breakdowns_to_test + "." + key
            # Determine the number of tester directories already present in the run directory
            tester_dirs = [
                d
                for d in os.listdir(run_base_dir.as_posix())
                if os.path.isdir(f"{run_base_dir}/{d}") and d.endswith("_tester")
            ]
            tester_number = len(tester_dirs) + 1
            # Format the tester number to have 2 digits
            tester_number_str = f"{tester_number:02}"
            user_name = f"{tester_number_str}_{key}_tester"
            dialogue_base_dir = run_base_dir / user_name
            os.makedirs(dialogue_base_dir, exist_ok=True)
            tester_instructions = bd.tester_instructions
            print(f"Tester instructions: {tester_instructions}")
            # Store tester info in a yaml file
            tester_info = {
                "run_id": run_id,
                "breakdown_key": full_breakdown_key,
                "title": bd.title,
                "description": bd.description,
                "tester_instructions": tester_instructions,
            }
            tester_info_file = f"{dialogue_base_dir}/info.yaml"
            with open(tester_info_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    tester_info, f, indent=4, sort_keys=False, allow_unicode=True
                )

            user_simulator = TestUserSimulator(
                bd,
                chatbot.info,
                model=user_simulator_llm,
                typical_user_turn_length=typical_user_turn_length,
                max_user_turn_length=max_user_turn_length,
                seed=seed,
            )
            user_simulator_setup_kwargs: dict = {}

            dialogues = simulate_dialogues(
                run_id,
                user_name,
                dialogue_base_dir,
                chatbot_client,
                user_simulator,
                user_simulator_setup_kwargs,
                max_user_turns,
                runs_per_user=runs_per_breakdown,
                save_prompt=save_prompt,
            )
            all_simulated_dialogues.extend(dialogues)
    return all_simulated_dialogues


def simulate_user_personas(
    run_id: str,
    chatbot: Chatbot,
    chatbot_client: ChatbotClientInterface,
    user_type: UserType,
    persona_id: Optional[str],
    max_user_messages: int,
    typical_user_turn_length: Optional[str] = None,
    max_user_turn_length: Optional[str] = None,
    runs_per_persona: int = 1,
    save_prompt=False,
    user_simulator_llm: str = DEFAULT_LLM,
    seed: Optional[int] = None,
) -> List[Dialogue]:
    available_user_personas = load_user_personas(chatbot)
    print(f"Available user personas: {available_user_personas}")

    personas_to_simulate: list[Persona] = []
    if persona_id:
        persona = available_user_personas.get(persona_id, None)
        if persona is None:
            error_str = f"User persona with ID {persona_id} not found."
            raise ValueError(error_str)
        personas_to_simulate.append(persona)
    else:
        if user_type == UserType.TESTERS:
            raise ValueError("Tester persona cannot be simulated in this function.")
        elif user_type == UserType.STANDARD_PERSONAS:
            personas_to_simulate = [
                persona
                for persona in available_user_personas.values()
                if persona.type == PersonaType.STANDARD
            ]
        elif user_type == UserType.CHALLENGING_PERSONAS:
            personas_to_simulate = [
                persona
                for persona in available_user_personas.values()
                if persona.type == PersonaType.CHALLENGING
            ]
        elif user_type == UserType.ADVERSARIAL_PERSONAS:
            personas_to_simulate = [
                persona
                for persona in available_user_personas.values()
                if persona.type == PersonaType.ADVERSARIAL
            ]
        elif user_type == UserType.ALL_PERSONAS:
            personas_to_simulate = list(available_user_personas.values())
        else:
            error_str = f"User type {user_type} not recognized."
            raise ValueError(error_str)

    print(
        f"Simulating {len(personas_to_simulate)} user personas for chatbot {chatbot.id}..."
    )
    all_simulated_dialogues = []
    for user_persona in tqdm(personas_to_simulate):
        current_persona_id: str = user_persona.persona_id
        print(f"Simulating user persona: {current_persona_id}")
        print(
            yaml.safe_dump(
                user_persona.model_dump(),
                indent=4,
                sort_keys=False,
                allow_unicode=True,
            )
        )
        dialogue_base_dir = (
            chatbot.base_directory / "runs" / run_id / current_persona_id
        )
        os.makedirs(dialogue_base_dir, exist_ok=True)
        # Store persona info in a yaml file
        persona_info_file = f"{dialogue_base_dir}/persona_info.yaml"
        persona_info = {"run_id": run_id, "persona": user_persona.model_dump()}
        with open(persona_info_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                persona_info, f, indent=4, sort_keys=False, allow_unicode=True
            )

        user_simulator = PersonaSimulator(
            user_persona,
            chatbot.info,
            model=user_simulator_llm,
            typical_user_turn_length=typical_user_turn_length,
            max_user_turn_length=max_user_turn_length,
            seed=seed,
        )
        user_simulator_setup_kwargs: dict = {}

        dialogues = simulate_dialogues(
            run_id,
            current_persona_id,
            dialogue_base_dir,
            chatbot_client,
            user_simulator,
            user_simulator_setup_kwargs,
            max_user_messages,
            runs_per_user=runs_per_persona,
            save_prompt=save_prompt,
        )
        all_simulated_dialogues.extend(dialogues)
    return all_simulated_dialogues


def run(
    chatbot: Chatbot,
    user_type: UserType,
    selector: Optional[str] = None,
    runs_per_user=1,
    run_prefix: Optional[str] = None,
    debug=True,
    seed: Optional[int] = None,
) -> str:
    test_run_id = f"{user_type}_{datetime.now().strftime('%Y-%m-%d')}_{datetime.now().strftime('%H-%M-%S')}"
    if seed is not None:
        test_run_id += f"_seed_{seed}"
    if run_prefix:
        test_run_id = f"{run_prefix}_{test_run_id}"
    else:
        test_run_id = f"run_{test_run_id}"
    print(f"Test run ID: {test_run_id}")

    typical_user_turn_length = chatbot.user_simulation_config.typical_user_turn_length
    max_user_turn_length = chatbot.user_simulation_config.max_user_turn_length
    max_user_turns = (
        chatbot.user_simulation_config.max_user_turns or DEFAULT_MAX_USER_TURNS
    )
    print(f"Max user turns set to: {max_user_turns}")

    print("Initializing chatbot...")

    client_module = SourceFileLoader(
        "chatbot_client", f"{chatbot.base_directory}/chatbot_client.py"
    ).load_module()
    chatbot_client: ChatbotClientInterface = client_module.ChatbotClient()
    chatbot_client.set_up_class()

    if user_type == UserType.AUTOTOD_MULTIWOZ_SCENARIOS:
        # For the AutoTOD-SIM we always use gpt-3.5-turbo-1106 based on the usage of gpt-3.5-turbo in the AutoTOD paper (https://github.com/DaDaMrX/AutoTOD)
        user_simulator_llm = "gpt-3.5-turbo-1106"
    else:
        user_simulator_llm = os.getenv("CHAT_CHECKER_USER_SIMULATOR_LLM", DEFAULT_LLM)

    run_info = {
        "run_id:": test_run_id,
        "chatbot_id": chatbot.id,
        "chatbot_info": chatbot.info.model_dump(),
        "user_type": user_type,
        "selector": selector,
        "runs_per_user": runs_per_user,
        "max_user_messages": max_user_turns,
        "typical_user_turn_length": typical_user_turn_length,
        "max_user_turn_length": max_user_turn_length,
        "debug": debug,
        "user_simulator_llm": user_simulator_llm,
        "seed": seed,
    }

    run_base_dir = chatbot.base_directory / "runs" / test_run_id
    os.makedirs(run_base_dir, exist_ok=True)
    run_info_file = run_base_dir / "simulation_run_info.yaml"
    with open(run_info_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_info, f, indent=4, sort_keys=False, allow_unicode=True)
    print(f"Run info saved to {run_info_file}")

    if user_type == UserType.TESTERS:
        all_simulated_dialogues = simulate_testers(
            test_run_id,
            chatbot,
            chatbot_client,
            selector or "",
            max_user_turns,
            typical_user_turn_length=typical_user_turn_length,
            max_user_turn_length=max_user_turn_length,
            runs_per_breakdown=runs_per_user,
            save_prompt=debug,
            user_simulator_llm=user_simulator_llm,
            seed=seed,
        )
    elif user_type == UserType.AUTOTOD_MULTIWOZ_SCENARIOS:
        all_simulated_dialogues = run_autotod_multiwoz_simulator(
            test_run_id,
            chatbot,
            chatbot_client,
            max_user_turns,
            n_dialogues=int(selector or 1),
            runs_per_user=runs_per_user,
            seed=seed,
        )
    elif user_type in [
        UserType.STANDARD_PERSONAS,
        UserType.CHALLENGING_PERSONAS,
        UserType.ADVERSARIAL_PERSONAS,
        UserType.ALL_PERSONAS,
    ]:
        all_simulated_dialogues = simulate_user_personas(
            test_run_id,
            chatbot,
            chatbot_client,
            user_type,
            selector or None,
            max_user_turns,
            typical_user_turn_length,
            max_user_turn_length=max_user_turn_length,
            runs_per_persona=runs_per_user,
            save_prompt=debug,
            user_simulator_llm=user_simulator_llm,
            seed=seed,
        )
    else:
        raise ValueError(f"User type {user_type} not recognized.")
    chatbot_client.tear_down_class()

    # Compute statistics
    run_stats = compute_run_statistics(all_simulated_dialogues)

    # Save the run statistics in the run info file
    run_info["chat_statistics"] = run_stats["run_chat_statistics"]
    run_info["simulation_cost_statistics"] = run_stats["run_cost_statistics"]
    with open(run_info_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_info, f, indent=4, sort_keys=False, allow_unicode=True)
    print(f"Run stats saved to {run_info_file}")

    print(f"Run {test_run_id} completed.")
    return test_run_id
