from pathlib import Path
import random
from typing import Annotated, Optional

import typer
from rich import print

from chat_checker.models.run import UserType
from chat_checker.models.user_personas import PersonaType
from chat_checker.data_management.chatbot_registry import register_chatbots, get_chatbot
from chat_checker.persona_generation.persona_generator import run as run_persona_gen
from chat_checker.simulation_runner import run as run_simulation
from chat_checker.breakdown_identification_runner import run as run_spot_errors
from chat_checker.rating_runner import run as run_evaluation
from chat_checker.utils.misc_utils import verify_environment

CHAT_CHECKER_BASE_DIR = Path(__file__).parent.parent

app = typer.Typer(
    help="chat-checker CLI for chatbot testing and evaluation",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# Definition of types for common arguments and options
RunID = Annotated[str, typer.Argument(..., help="ID of the simulation run to analyze")]
ChatbotID = Annotated[
    str,
    typer.Argument(..., help="ID of the chatbot for which the command should be run"),
]
ChatbotsBaseDir = Annotated[
    Path,
    typer.Option(
        "--chatbots-base-dir",
        "-d",
        help="Base directory containing the chatbot configurations, client implementations, and runs",
    ),
]
UserTypeSel = Annotated[
    UserType,
    typer.Option(
        "--user-type",
        "-u",
        help="Type of users to simulate",
    ),
]
Selector = Annotated[
    Optional[str],
    typer.Option(
        "--selector",
        "-sel",
        help="Selector for the specific user personas to simulate. For persona-based user types, this can be the persona ID. For breakdown-testers this is the selector for the breakdowns to test (e.g. 'task_oriented.task_success_failures'). For autotod_multiwoz, this is the number of dialogues to simulate. All users of the given type are simulated if not provided",
    ),
]
RunsPerUser = Annotated[
    int,
    typer.Option("--runs-per-user", "-r", help="Number of runs per user to simulate"),
]
RunPrefix = Annotated[
    Optional[str],
    typer.Option("--run-prefix", "-rp", help="Prefix for the run ID"),
]
Subfolder = Annotated[
    Optional[str],
    typer.Option(
        "--subfolder",
        "-s",
        help="Subfolder containing the subset of dialogues to analyze. All dialogues of the run are analyzed if not provided",
    ),
]
DialogueFileName = Annotated[
    Optional[str],
    typer.Option(
        "--file",
        "-f",
        help="Name of the dialogue file to analyze. All dialogues of the run are analyzed if not provided",
    ),
]
ExtraOutputFile = Annotated[
    bool,
    typer.Option(
        "--extra-output",
        "-e",
        help="Store the analysis annotations in an extra output file separate from the original dialogue file",
    ),
]
RecomputeStats = Annotated[
    bool,
    typer.Option(
        "--recompute-stats",
        "-rc",
        help="Recompute statistics for the existing analysis, don't analyze again",
    ),
]

Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose mode")]
Debug = Annotated[bool, typer.Option("--debug", "-d", help="Enable debug mode")]
Seed = Annotated[
    Optional[int],
    typer.Option("--seed", "-s", help="Seed for the random number generator"),
]


@app.command()
def register(
    chatbots_base_dir: ChatbotsBaseDir = Path("./chatbots"),
    chatbot_id: Optional[str] = typer.Option(
        None,
        "--chatbot-id",
        "-c",
        help="ID of the chatbot within the directory to register. Otherwise, all chatbots found in the directory are registered",
    ),
):
    """
    Register chatbots contained in the base directory for testing and evaluation.
    """
    valid_env = verify_environment(is_cli=True)
    if not valid_env:
        return
    register_chatbots(chatbots_base_dir, chatbot_id)


@app.command()
def generate_personas(
    chatbot_id: ChatbotID,
    persona_type: PersonaType = typer.Option(
        PersonaType.STANDARD, "--type", "-t", help="Type of personas to generate"
    ),
    num_personas: int = typer.Option(
        1, "--num", "-n", help="Number of personas to generate"
    ),
    verbose: Verbose = False,
    seed: Seed = None,
):
    """
    Generate user personas for chatbot simulation.
    """
    valid_env = verify_environment(is_cli=True)
    if not valid_env:
        return
    try:
        chatbot = get_chatbot(chatbot_id)
    except ValueError as e:
        print(e)
        return
    if seed is not None:
        # set the seed for the random number generator
        random.seed(seed)
    run_persona_gen(
        chatbot=chatbot,
        persona_type=persona_type,
        num_personas=num_personas,
        verbose=verbose,
        seed=seed,
    )


@app.command()
def simulate_users(
    chatbot_id: ChatbotID,
    user_type: UserTypeSel = UserType.ALL_PERSONAS,
    selector: Selector = None,
    runs_per_user: RunsPerUser = 1,
    run_prefix: RunPrefix = None,
    debug: Debug = False,
    seed: Seed = None,
):
    """
    Simulate users interacting with a chatbot.
    """
    valid_env = verify_environment(is_cli=True)
    if not valid_env:
        return
    try:
        chatbot = get_chatbot(chatbot_id)
    except ValueError as e:
        print(e)
        return
    if seed is not None:
        # set the seed for the random number generator
        random.seed(seed)
    run_simulation(
        chatbot=chatbot,
        user_type=user_type,
        selector=selector,
        runs_per_user=runs_per_user,
        run_prefix=run_prefix,
        debug=debug,
        seed=seed,
    )


@app.command()
def test(
    chatbot_id: ChatbotID,
    run_id: RunID,
    subfolder: Subfolder = None,
    dialogue_file_name: DialogueFileName = None,
    extra_output_file: ExtraOutputFile = False,
    recompute_stats: RecomputeStats = False,
    seed: Seed = None,
):
    """
    Run tests to spot errors in dialogues from a previous run.
    """
    if dialogue_file_name and not subfolder:
        raise typer.BadParameter(
            "If providing a dialogue file name, you must also provide a subfolder"
        )
    valid_env = verify_environment(is_cli=True)
    if not valid_env:
        return
    try:
        chatbot = get_chatbot(chatbot_id)
    except ValueError as e:
        print(e)
        return
    if seed is not None:
        # set the seed for the random number generator
        random.seed(seed)
    run_spot_errors(
        chatbot=chatbot,
        run_id=run_id,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        extra_output_file=extra_output_file,
        recompute_stats=recompute_stats,
        save_prompts=True,
        seed=seed,
    )


@app.command()
def evaluate(
    chatbot_id: ChatbotID,
    run_id: RunID,
    subfolder: Subfolder = None,
    dialogue_file_name: DialogueFileName = None,
    extra_output_file: ExtraOutputFile = False,
    recompute_stats: RecomputeStats = False,
    seed: Seed = None,
):
    """
    Evaluate dialogues from a previous run.
    """
    if dialogue_file_name and not subfolder:
        raise typer.BadParameter(
            "If providing a dialogue file name, you must also provide a subfolder"
        )

    valid_env = verify_environment(is_cli=True)
    if not valid_env:
        return
    try:
        chatbot = get_chatbot(chatbot_id)
    except ValueError as e:
        print(e)
        return

    run_evaluation(
        chatbot=chatbot,
        run_id=run_id,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        extra_output_file=extra_output_file,
        recompute_stats=recompute_stats,
        seed=seed,
    )


@app.command()
def run(
    chatbot_id: ChatbotID,
    user_type: UserTypeSel = UserType.ALL_PERSONAS,
    selector: Selector = None,
    runs_per_user: RunsPerUser = 1,
    run_prefix: RunPrefix = None,
    subfolder: Subfolder = None,
    dialogue_file_name: DialogueFileName = None,
    extra_output_file: ExtraOutputFile = False,
    recompute_stats: RecomputeStats = False,
    debug: Debug = False,
    seed: Seed = None,
):
    """
    Run the full pipeline: simulate users, spot errors, and evaluate dialogues.
    """
    valid_env = verify_environment(is_cli=True)
    if not valid_env:
        return
    try:
        chatbot = get_chatbot(chatbot_id)
    except ValueError as e:
        print(e)
        return
    print(f"Running full pipeline for chatbot {chatbot_id}")

    if seed is not None:
        # set the seed for the random number generator
        random.seed(seed)

    # Step 1: Simulate users
    print("Step 1: Simulating users")
    run_id = run_simulation(
        chatbot=chatbot,
        user_type=user_type,
        selector=selector,
        runs_per_user=runs_per_user,
        run_prefix=run_prefix,
        debug=debug,
        seed=seed,
    )

    # Step 2: Spot errors
    print(f"Step 2: Spotting errors in run {run_id}")
    run_spot_errors(
        chatbot=chatbot,
        run_id=run_id,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        extra_output_file=extra_output_file,
        recompute_stats=recompute_stats,
        seed=seed,
    )

    # Step 3: Evaluate dialogues
    print(f"Step 3: Evaluating dialogues from run {run_id}")
    run_evaluation(
        chatbot=chatbot,
        run_id=run_id,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        extra_output_file=extra_output_file,
        recompute_stats=recompute_stats,
        seed=seed,
    )

    print("Full pipeline completed successfully")


if __name__ == "__main__":
    app()
