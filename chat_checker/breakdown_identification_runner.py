import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from chat_checker.breakdown_detection.breakdown_detector import find_dialogue_breakdowns
from chat_checker.breakdown_detection.breakdown_taxonomy import get_flattened_taxonomy
from chat_checker.data_management.storage_manager import load_dialogues
from chat_checker.models.breakdowns import BreakdownDecision
from chat_checker.models.chatbot import Chatbot, ChatbotType
from chat_checker.models.dialogue import Dialogue, DialogueTurn, SpeakerRole
from chat_checker.models.llm import UsageCost
from chat_checker.utils.llm_utils import compute_total_usage, DEFAULT_LLM
from chat_checker.utils.misc_utils import (
    compute_analysis_cost_statistics,
    five_num_summary,
)

# Build the path to the .env file
BASE_DIR = Path(__file__).parent

# set tick and label font size
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["legend.fontsize"] = "large"


def compute_dialogue_breakdown_stats(
    dialogue_start_time: datetime,
    dialogue_end_time: datetime,
    breakdown_detection_usage: UsageCost,
    chat_history: List[DialogueTurn],
    dialogue: Dialogue,
    is_task_oriented: bool,
) -> None:
    turns_with_breakdowns: list[DialogueTurn] = []
    for turn in chat_history:
        if (
            turn.breakdown_annotation
            and turn.breakdown_annotation.decision == BreakdownDecision.BREAKDOWN
        ):
            turns_with_breakdowns.append(turn)
    breakdown_count = len(turns_with_breakdowns)
    system_turns = [
        turn for turn in chat_history if turn.role == SpeakerRole.DIALOGUE_SYSTEM
    ]
    if len(system_turns) == 0:
        avg_score = 0.0
    else:
        scores: list[float] = []
        for turn in system_turns:
            if turn.breakdown_annotation:
                scores.append(turn.breakdown_annotation.score)
        avg_score = sum(scores if scores else [0.0]) / len(system_turns)
    breakdown_turn_ids = [turn.turn_id for turn in turns_with_breakdowns]
    counts_per_type: Dict[str, int] = {}
    for key, value in get_flattened_taxonomy(is_task_oriented).items():
        breakdowns_for_key = []
        for turn in system_turns:
            target_error_type = value.title.lower()
            predicted_error_types = (
                [
                    error_type.lower()
                    for error_type in turn.breakdown_annotation.breakdown_types
                ]
                if turn.breakdown_annotation
                else []
            )
            if target_error_type in predicted_error_types:
                breakdowns_for_key.append(turn)

        counts_for_key = len(breakdowns_for_key)
        counts_per_type[key] = counts_for_key
    crash_turns = []
    for turn in system_turns:
        if (
            turn.breakdown_annotation
            and "Chatbot Crash" in turn.breakdown_annotation.breakdown_types
            and turn.breakdown_annotation.decision == BreakdownDecision.BREAKDOWN
        ):
            crash_turns.append(turn)
    counts_per_type["chatbot_crash"] = len(crash_turns)

    dialogue.breakdown_stats = {
        "analysis_start_time": dialogue_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_end_time": dialogue_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": breakdown_count,
        "avg_score": avg_score,
        "turn_ids_of_breakdowns": breakdown_turn_ids,
        "counts_per_breakdown_type": counts_per_type,
    }
    dialogue.breakdown_stats["detection_cost_stats"] = (
        breakdown_detection_usage.model_dump()
    )

    print(f"Number of detected breakdowns: {breakdown_count}")
    print(f"Average score: {avg_score}")
    print(
        f"Types of breakdowns: {[key for key, value in counts_per_type.items() if value > 0]}"
    )
    print(f"Turn IDs of breakdowns: {breakdown_turn_ids}")


def plot_and_save_heatmap(
    dialogues: list[Dialogue], dialogues_dir: Path
) -> Dict[str, Dict[str, int]]:
    # Compute a heatmap of the breakdown types per simulated user (dialogues of each user are aggregated)
    heatmap: Dict[str, Dict[str, int]] = {}
    flattened_taxonomy = get_flattened_taxonomy(True)
    for dialogue in dialogues:
        user_name = dialogue.user_name
        counts_per_type_by_key = (
            dialogue.breakdown_stats.get("counts_per_breakdown_type", {})
            if dialogue.breakdown_stats
            else {}
        )
        # replace the breakdown type keys with the breakdown titles
        counts_per_type_by_title = {
            (
                flattened_taxonomy[key].title
                if key != "chatbot_crash"
                else "Chatbot crash"
            ): value
            for key, value in counts_per_type_by_key.items()
        }
        if user_name not in heatmap:
            heatmap[user_name] = counts_per_type_by_key
        else:
            # Add the counts per breakdown type to the existing counts for the user
            for key, value in counts_per_type_by_key.items():
                if key in heatmap[user_name]:
                    heatmap[user_name][key] += value
                else:
                    heatmap[user_name][key] = value
    # plot the heatmap with matplotlib (x-axis: dialogue IDs, y-axis: breakdown types)
    if len(heatmap) > 0:
        fig, ax = plt.subplots(layout="constrained")

        heatmap_data = []
        for _, counts_per_type_by_key in heatmap.items():
            row = []
            for key, value in reversed(counts_per_type_by_key.items()):
                row.append(value)
            heatmap_data.append(row)
        # transpose the heatmap data for plotting (cf. https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html for the expected data format)
        heatmap_data = list(map(list, zip(*heatmap_data)))
        im = ax.imshow(heatmap_data)
        ax.set_xticks(range(len(heatmap.keys())))
        ax.set_yticks(range(len(counts_per_type_by_title.keys())))
        ax.set_xticklabels(heatmap.keys(), rotation=45, ha="right")
        ax.set_yticklabels(reversed(counts_per_type_by_title.keys()))

        # Create a colorbar
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(im, cax=colorbar_ax)
        cbar.ax.set_ylabel(
            "Number of breakdowns", rotation=90, va="top", fontsize="large"
        )

        # Make figure bigger and give more space for the x-axis labels
        fig.set_size_inches(14, 12)
        # fig.tight_layout()

        # save the heatmap
        plt.savefig(dialogues_dir / "breakdown_heatmap.png")
        print(
            f"Heatmap of breakdown types saved to {dialogues_dir / 'breakdown_heatmap.png'}"
        )
    return heatmap


def compute_breakdown_matches_per_test_user(
    heatmap: Dict[str, Dict[str, int]],
) -> Tuple[float, str, List[str]]:
    # Compute how often the index of the breakdown type matches the index of the test user
    breakdown_matches = 0
    users_with_matches = []
    for i, (user_name, counts_per_type) in enumerate(heatmap.items()):
        for j, (breakdown_key, value) in enumerate(counts_per_type.items()):
            if breakdown_key in user_name and value > 0:
                breakdown_matches += 1
                users_with_matches.append(user_name)
    return (
        breakdown_matches / len(heatmap),
        f"{breakdown_matches}/{len(heatmap)}",
        users_with_matches,
    )


def compute_run_breakdown_stats(
    analysis_start_time: datetime,
    analysis_end_time: datetime,
    dialogues: list[Dialogue],
    total_breakdown_detection_usage: UsageCost,
    is_task_oriented: bool,
    dialogues_dir: Path,
    chatbot_id: Optional[str] = None,
    real_dialogue: bool = False,
    run_id: Optional[str] = None,
    subfolder: Optional[str] = None,
    dialogue_file_name: Optional[str] = None,
    extra_output_file: bool = False,
) -> None:
    dialogues_with_breakdowns = [
        dialogue
        for dialogue in dialogues
        if dialogue.breakdown_stats and dialogue.breakdown_stats.get("count", 0) > 0
    ]
    num_chatbot_turns = sum(
        [
            dialogue.chat_statistics["num_chatbot_turns"]
            for dialogue in dialogues
            if dialogue.chat_statistics
        ]
    )
    breakdown_counts = [
        dialogue.breakdown_stats.get("count", 0)
        for dialogue in dialogues
        if dialogue.breakdown_stats
    ]
    avg_scores = [
        dialogue.breakdown_stats.get("avg_score", 0)
        for dialogue in dialogues
        if dialogue.breakdown_stats
    ]

    total_breakdown_count = sum(breakdown_counts)
    total_avg_score = sum(avg_scores) / len(avg_scores) if len(avg_scores) > 0 else None
    print(f"Total number of detected breakdowns: {total_breakdown_count}")
    print(f"Average score: {total_avg_score}")
    print(
        f"Dialogues with breakdowns: {[dialogue.dialogue_id for dialogue in dialogues_with_breakdowns]}"
    )

    ids_of_first_breakdowns = [
        dialogue.breakdown_stats.get("turn_ids_of_breakdowns", [])[0]
        for dialogue in dialogues_with_breakdowns
        if dialogue.breakdown_stats
    ]
    avg_turn_number_of_first_breakdown = (
        sum(ids_of_first_breakdowns) / len(ids_of_first_breakdowns)
        if len(ids_of_first_breakdowns) > 0
        else None
    )

    turns_with_breakdowns: list[DialogueTurn] = []
    for dialogue in dialogues_with_breakdowns:
        chat_history = dialogue.chat_history
        for turn in chat_history:
            if (
                turn.breakdown_annotation
                and turn.breakdown_annotation.decision == BreakdownDecision.BREAKDOWN
            ):
                turns_with_breakdowns.append(turn)

    scores_of_turns_with_breakdowns = [
        turn.breakdown_annotation.score
        for turn in turns_with_breakdowns
        if turn.breakdown_annotation
    ]

    scores_of_turns_with_breakdowns_excluding_chatbot_crashes = []
    for turn in turns_with_breakdowns:
        if turn.breakdown_annotation and turn.breakdown_annotation.breakdown_types != [
            "Chatbot Crash"
        ]:
            scores_of_turns_with_breakdowns_excluding_chatbot_crashes.append(
                turn.breakdown_annotation.score
            )

    five_num_summary_of_breakdown_scores = five_num_summary(
        scores_of_turns_with_breakdowns
    )
    five_num_summary_of_breakdown_scores_excluding_chatbot_crashes = five_num_summary(
        scores_of_turns_with_breakdowns_excluding_chatbot_crashes
    )

    # Compute the counts per breakdown type
    counts_per_type = {}
    for key, value in get_flattened_taxonomy(is_task_oriented).items():
        counts_for_key = sum(
            [
                dialogue.breakdown_stats["counts_per_breakdown_type"].get(key, 0)
                for dialogue in dialogues_with_breakdowns
                if dialogue.breakdown_stats
            ]
        )
        counts_per_type[key] = counts_for_key
    counts_per_type["chatbot_crash"] = sum(
        [
            dialogue.breakdown_stats["counts_per_breakdown_type"].get(
                "chatbot_crash", 0
            )
            for dialogue in dialogues_with_breakdowns
            if dialogue.breakdown_stats
        ]
    )
    n_unique_breakdown_types = len(
        [key for key, value in counts_per_type.items() if value > 0]
    )

    # Create breakdown excerpts for all dialogues with breakdowns
    breakdown_excerpts = []
    for dialogue in dialogues_with_breakdowns:
        chat_history = dialogue.chat_history
        for i, turn in enumerate(chat_history):
            if (
                turn.breakdown_annotation
                and turn.breakdown_annotation.decision == BreakdownDecision.BREAKDOWN
            ):
                previous_turn = chat_history[i - 1] if i > 0 else None
                excerpt = {
                    "dialogue_id": dialogue.dialogue_id,
                    "previous_turn": previous_turn.model_dump()
                    if previous_turn
                    else None,
                    "breakdown_turn": turn.model_dump(),
                }
                breakdown_excerpts.append(excerpt)

    heatmap = plot_and_save_heatmap(dialogues, dialogues_dir)
    breakdown_matches, breakdown_matches_str, users_with_matches = (
        compute_breakdown_matches_per_test_user(heatmap)
    )

    cost_stats = compute_analysis_cost_statistics(
        dialogues, total_breakdown_detection_usage
    )

    # Save the overall statistics
    test_run_info: dict[str, Any] = {
        "chatbot_id": chatbot_id,
        "real_dialogue": real_dialogue,
        "run_id": run_id,
        "subfolder": subfolder,
        "dialogue_file": dialogue_file_name,
        "extra_output_file": extra_output_file,
        "stats": {
            "start_time": analysis_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": analysis_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_analyzed_dialogues": len(dialogues),
            "n_dialogues_with_breakdowns": len(dialogues_with_breakdowns),
            "total_breakdown_count": total_breakdown_count,
            "n_analyzed_chatbot_turns": num_chatbot_turns,
            "breakdowns_per_chatbot_turn": total_breakdown_count / num_chatbot_turns
            if num_chatbot_turns > 0
            else None,
            "avg_turn_number_of_first_breakdown": avg_turn_number_of_first_breakdown,
            "avg_turn_quality_score": total_avg_score,
            "scores_of_turns_with_breakdowns": five_num_summary_of_breakdown_scores,
            "scores_of_turns_with_breakdowns_excluding_chatbot_crashes": five_num_summary_of_breakdown_scores_excluding_chatbot_crashes,
            "dialogues_with_breakdowns": [
                dialogue.dialogue_id for dialogue in dialogues_with_breakdowns
            ],
            "counts_per_breakdown_type": counts_per_type,
            "n_unique_breakdown_types": n_unique_breakdown_types,
            "breakdown_matches_per_user": breakdown_matches,
            "breakdown_matches_per_user_str": breakdown_matches_str,
            "users_with_matches": users_with_matches,
            "detection_cost_stats": cost_stats,
        },
        "breakdown_excerpts": breakdown_excerpts,
    }

    test_run_info_path = dialogues_dir / "breakdown_detection_stats.yaml"
    with open(test_run_info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(test_run_info, f, indent=4, sort_keys=False, allow_unicode=True)
    print(f"Aggregated statistics saved to {test_run_info_path}")


def test_dialogues(
    run_id: str,
    dialogues_dir: Path,
    dialogues: List[Dialogue],
    chatbot: Chatbot,
    is_task_oriented: bool = True,
    real_dialogue: bool = False,
    subfolder: Optional[str] = None,
    dialogue_file_name: Optional[str] = None,
    save_prompts: bool = True,
    extra_output_file: bool = False,
    recompute_stats: bool = False,
    breakdown_detector_model: str = DEFAULT_LLM,
    seed: Optional[int] = None,
):
    if recompute_stats:
        # Load the existing breakdown_detection_stats.yaml file
        breakdown_detection_stats_file = (
            dialogues_dir / "breakdown_detection_stats.yaml"
        )
        if not breakdown_detection_stats_file.exists():
            raise ValueError(
                f"Can not recompute stats, as the file {breakdown_detection_stats_file} does not exist."
            )
        with open(breakdown_detection_stats_file, "r", encoding="utf-8") as f:
            existing_breakdown_detection_stats = yaml.safe_load(f)

        analysis_start_time = datetime.strptime(
            existing_breakdown_detection_stats["stats"]["start_time"],
            "%Y-%m-%d %H:%M:%S",
        )
        print(
            f"Recomputing breakdown detection statistics for {len(dialogues)} dialogues..."
        )
    else:
        analysis_start_time = datetime.now()
        print(f"Analyzing {len(dialogues)} dialogues...")
    total_breakdown_detection_usage = UsageCost(
        prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
    )
    for i, dialogue in enumerate(dialogues):
        print(
            f"Analyzing dialogue {dialogue.dialogue_id} ({i + 1}/{len(dialogues)})..."
        )

        chat_history = dialogue.chat_history

        if recompute_stats:
            if not dialogue.breakdown_stats:
                raise ValueError(
                    f"Can not run in stats_only mode, as the dialogue {dialogue.path} does not have breakdown statistics."
                )
            detection_start_time = datetime.strptime(
                dialogue.breakdown_stats["analysis_start_time"], "%Y-%m-%d %H:%M:%S"
            )
            detection_end_time = datetime.strptime(
                dialogue.breakdown_stats["analysis_end_time"], "%Y-%m-%d %H:%M:%S"
            )
            breakdown_detection_usage = UsageCost(
                **dialogue.breakdown_stats["detection_cost_stats"]
            )
        else:
            detection_start_time = datetime.now()
            model_responses = find_dialogue_breakdowns(
                chat_history,
                is_task_oriented,
                chatbot.info,
                save_prompts=save_prompts,
                save_dir=(dialogue.path.parent / "breakdown_detection_prompts"),
                breakdown_detector_model=breakdown_detector_model,
                seed=seed,
            )
            detection_end_time = datetime.now()
            breakdown_detection_usage = compute_total_usage(model_responses)

        total_breakdown_detection_usage.prompt_tokens += (
            breakdown_detection_usage.prompt_tokens
        )
        total_breakdown_detection_usage.completion_tokens += (
            breakdown_detection_usage.completion_tokens
        )
        total_breakdown_detection_usage.total_tokens += (
            breakdown_detection_usage.prompt_tokens
            + breakdown_detection_usage.completion_tokens
        )
        total_breakdown_detection_usage.cost += breakdown_detection_usage.cost

        compute_dialogue_breakdown_stats(
            detection_start_time,
            detection_end_time,
            breakdown_detection_usage,
            chat_history,
            dialogue,
            is_task_oriented,
        )

        if extra_output_file:
            output_path = dialogue.path.parent / f"{dialogue.path.stem}_annotated.yaml"
        else:
            output_path = dialogue.path
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                dialogue.model_dump(), f, indent=4, sort_keys=False, allow_unicode=True
            )

        print(f"Annotated dialogue saved to {output_path}")

    analysis_end_time = (
        datetime.now()
        if not recompute_stats
        else datetime.strptime(
            existing_breakdown_detection_stats["stats"]["end_time"], "%Y-%m-%d %H:%M:%S"
        )
    )
    print(
        f"Breakdown detection for {len(dialogues)} dialogues completed. Aggregating statistics..."
    )
    compute_run_breakdown_stats(
        analysis_start_time,
        analysis_end_time,
        dialogues,
        total_breakdown_detection_usage,
        is_task_oriented,
        dialogues_dir,
        chatbot_id=chatbot.id,
        real_dialogue=real_dialogue,
        run_id=run_id,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        extra_output_file=extra_output_file,
    )
    print("Analysis completed.")


def run(
    chatbot: Chatbot,
    run_id: str,
    subfolder: Optional[str] = None,
    dialogue_file_name: Optional[str] = None,
    real_dialogue: bool = False,
    extra_output_file: bool = False,
    recompute_stats: bool = False,
    save_prompts: bool = True,
    seed: Optional[int] = None,
):
    if dialogue_file_name and not subfolder:
        raise ValueError(
            "If a dialogue file is specified, a subfolder must also be specified."
        )

    is_task_oriented = chatbot.info.type == ChatbotType.TASK_ORIENTED

    dialogues_dir, dialogues = load_dialogues(
        chatbot.base_directory, run_id, subfolder, dialogue_file_name, real_dialogue
    )

    if not dialogues:
        print(f"No dialogues found to analyze in {dialogues_dir}. Exiting...")
        return

    breakdown_detector_model = os.getenv(
        "CHAT_CHECKER_BREAKDOWN_DETECTOR_LLM", DEFAULT_LLM
    )

    test_dialogues(
        run_id,
        dialogues_dir,
        dialogues,
        chatbot,
        is_task_oriented=is_task_oriented,
        real_dialogue=real_dialogue,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        extra_output_file=extra_output_file,
        recompute_stats=recompute_stats,
        save_prompts=save_prompts,
        breakdown_detector_model=breakdown_detector_model,
        seed=seed,
    )
