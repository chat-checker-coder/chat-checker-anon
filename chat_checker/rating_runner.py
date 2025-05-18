import os
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime

import numpy as np
import yaml
from lexical_diversity import lex_div


from chat_checker.data_management.storage_manager import load_dialogues
from chat_checker.dialogue_rating.dialogue_rater import (
    get_dialogue_rating,
)
from chat_checker.models.chatbot import Chatbot
from chat_checker.models.dialogue import Dialogue, SpeakerRole
from chat_checker.models.llm import UsageCost
from chat_checker.utils.llm_utils import DEFAULT_LLM, compute_total_usage
from chat_checker.utils.misc_utils import (
    compute_analysis_cost_statistics,
    five_num_summary,
)


def compute_run_evaluation_stats(
    analysis_start_time: datetime,
    analysis_end_time: datetime,
    dialogues: List[Dialogue],
    total_evaluation_usage: UsageCost,
    chatbot: Chatbot,
    dialogues_dir: Path,
    real_dialogue: bool = False,
    run_id: Optional[str] = None,
    subfolder: Optional[str] = None,
    dialogue_file_name: Optional[str] = None,
    extra_output_file: bool = False,
):
    # Compute averages, std and five number summaries for the individual rating_dimensions
    rating_stats = {}

    for rating_dimension in chatbot.rating_dimensions:
        ratings = [
            dialogue.ratings[rating_dimension.key].rating
            for dialogue in dialogues
            if dialogue.ratings
        ]
        avg_rating = float(np.mean(ratings))
        std_rating = float(np.std(ratings))
        five_number_summary = five_num_summary(ratings)

        rating_dimension_stats = {
            "average": avg_rating,
            "std": std_rating,
            "five_number_summary": five_number_summary,
        }

        rating_stats[rating_dimension.key] = rating_dimension_stats

    cost_stats = compute_analysis_cost_statistics(dialogues, total_evaluation_usage)

    all_user_turns = [
        turn
        for dialogue in dialogues
        for turn in dialogue.chat_history
        if turn.role == SpeakerRole.USER
    ]
    all_chatbot_turns = [
        turn
        for dialogue in dialogues
        for turn in dialogue.chat_history
        if turn.role == SpeakerRole.DIALOGUE_SYSTEM
    ]
    all_user_turn_tokens = []
    for turn in all_user_turns:
        tokens = lex_div.tokenize(turn.content)
        all_user_turn_tokens.extend(tokens)
    all_chatbot_turn_tokens = []
    for turn in all_chatbot_turns:
        tokens = lex_div.tokenize(turn.content)
        all_chatbot_turn_tokens.extend(tokens)

    evaluation_run_info: dict[str, Any] = {
        "chatbot_id": chatbot.id,
        "real_dialogue": real_dialogue,
        "run_id": run_id,
        "subfolder": subfolder,
        "dialogue_file": dialogue_file_name,
        "extra_output_file": extra_output_file,
        "stats": {
            "start_time": analysis_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": analysis_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_analyzed_dialogues": len(dialogues),
            "user_turn_mtld": lex_div.mtld(all_user_turn_tokens),
            "chatbot_turn_mtld": lex_div.mtld(all_chatbot_turn_tokens),
            "rating_stats": rating_stats,
            "cost_stats": cost_stats,
        },
    }

    evaluation_run_info_path = dialogues_dir / "evaluation_stats.yaml"
    with open(evaluation_run_info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            evaluation_run_info, f, indent=4, sort_keys=False, allow_unicode=True
        )
    print(f"Aggregated statistics saved to {evaluation_run_info_path}")


def evaluate_dialogues(
    run_id: str,
    dialogues_dir: Path,
    dialogues: list[Dialogue],
    chatbot: Chatbot,
    real_dialogue: bool = False,
    subfolder: Optional[str] = None,
    dialogue_file_name: Optional[str] = None,
    save_prompts: bool = False,
    extra_output_file: bool = False,
    stats_only: bool = False,
    rating_model: str = DEFAULT_LLM,
    seed: Optional[int] = None,
):
    if stats_only:
        rating_stats_file = dialogues_dir / "evaluation_stats.yaml"
        if not rating_stats_file.exists():
            raise ValueError(
                f"Can not run in stats_only mode, as the file {rating_stats_file} does not exist."
            )
        with open(rating_stats_file, "r", encoding="utf-8") as f:
            existing_rating_stats = yaml.safe_load(f)

        analysis_start_time = datetime.strptime(
            existing_rating_stats["stats"]["start_time"],
            "%Y-%m-%d %H:%M:%S",
        )
        print(f"Recomputing evaluation statistics for {len(dialogues)} dialogues...")
    else:
        analysis_start_time = datetime.now()
        print(f"Analyzing {len(dialogues)} dialogues...")
    total_eval_usage = UsageCost(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost=0,
    )

    rated_dialogues = []
    for i, dialogue in enumerate(dialogues):
        print(
            f"Analyzing dialogue {dialogue.dialogue_id} ({i + 1}/{len(dialogues)})..."
        )

        chat_history = dialogue.chat_history

        if stats_only:
            if not dialogue.eval_stats:
                raise ValueError(
                    f"Can not run in stats_only mode, as the dialogue {dialogue} does not have breakdown statistics."
                )
            eval_start_time = datetime.strptime(
                dialogue.eval_stats["evaluation_start_time"], "%Y-%m-%d %H:%M:%S"
            )
            eval_end_time = datetime.strptime(
                dialogue.eval_stats["evaluation_end_time"], "%Y-%m-%d %H:%M:%S"
            )
            eval_usage = UsageCost(**dialogue.eval_stats["cost_stats"])
        else:
            eval_start_time = datetime.now()
            rating, messages, model_response = get_dialogue_rating(
                chat_history,
                rating_dimensions=chatbot.rating_dimensions,
                chatbot_info=chatbot.info,
                rating_model=rating_model,
                seed=seed,
            )
            eval_end_time = datetime.now()
            eval_usage = compute_total_usage([model_response])
            if save_prompts:
                prompt_str = "\n\n".join(
                    [f"{message['role']}: {message['content']}" for message in messages]
                )
                save_dir = dialogue.path.parent / "evaluation_prompts"
                save_dir.mkdir(exist_ok=True)
                with open(
                    save_dir / f"{dialogue.path.stem}_prompt.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(prompt_str)
            dialogue.ratings = rating

        dialogue.eval_stats = {
            "evaluation_start_time": eval_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_end_time": eval_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "cost_stats": eval_usage.model_dump(),
        }

        total_eval_usage.prompt_tokens += eval_usage.prompt_tokens
        total_eval_usage.completion_tokens += eval_usage.completion_tokens
        total_eval_usage.total_tokens += eval_usage.total_tokens
        total_eval_usage.cost += eval_usage.cost
        if extra_output_file:
            output_path = dialogue.path.parent / f"{dialogue.path.stem}_annotated.yaml"
        else:
            output_path = dialogue.path
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                dialogue.model_dump(), f, indent=4, sort_keys=False, allow_unicode=True
            )

        print(f"Rated dialogue saved to {output_path}")

        rated_dialogues.append(dialogue)

    analysis_end_time = datetime.now()

    print(
        f"Evaluation completed for {len(rated_dialogues)} dialogues. Aggregating statistics..."
    )

    compute_run_evaluation_stats(
        analysis_start_time,
        analysis_end_time,
        rated_dialogues,
        total_eval_usage,
        chatbot,
        dialogues_dir,
        real_dialogue,
        run_id,
        subfolder,
        dialogue_file_name,
        extra_output_file,
    )
    print("Analysis completed")


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

    dialogues_dir, dialogues = load_dialogues(
        chatbot.base_directory, run_id, subfolder, dialogue_file_name, real_dialogue
    )

    if not dialogues:
        print(f"No dialogues found to analyze in {dialogues_dir}. Exiting...")
        return

    rating_model = os.getenv("CHAT_CHECKER_DIALOGUE_RATER_LLM", DEFAULT_LLM)

    evaluate_dialogues(
        run_id,
        dialogues_dir,
        dialogues,
        chatbot,
        real_dialogue=real_dialogue,
        subfolder=subfolder,
        dialogue_file_name=dialogue_file_name,
        save_prompts=save_prompts,
        extra_output_file=extra_output_file,
        stats_only=recompute_stats,
        rating_model=rating_model,
        seed=seed,
    )
