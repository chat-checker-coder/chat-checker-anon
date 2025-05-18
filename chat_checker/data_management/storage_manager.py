import os
from pathlib import Path
from typing import Optional

import yaml

from chat_checker.models.chatbot import Chatbot
from chat_checker.models.dialogue import Dialogue
from chat_checker.models.user_personas import Persona


def load_dialogues(
    chatbot_base_dir: Path,
    run_id: str,
    subfolder: Optional[str] = None,
    dialogue_file_name: Optional[str] = None,
    real_dialogue: bool = False,
) -> tuple[Path, list[Dialogue]]:
    if real_dialogue:
        dialogues_dir = chatbot_base_dir / "real_dialogues"
    else:
        dialogues_dir = chatbot_base_dir / "runs" / run_id
    if subfolder:
        dialogues_dir = dialogues_dir / subfolder

    if dialogue_file_name:
        # Find the dialogue file in the specified directory and subdirectories
        dialogue_files = [
            f for f in dialogues_dir.glob("**/*.yaml") if f.stem == dialogue_file_name
        ]
        if not dialogue_files:
            raise FileNotFoundError(
                f"Could not find dialogue file {dialogue_file_name}.yaml in the specified directory and subdirectories."
            )
    else:
        # Find all dialogues in the specified directory and subdirectories
        dialogue_files = [
            f
            for f in dialogues_dir.glob("**/*.yaml")
            if "dialogue" in f.stem and not f.stem.endswith("_annotated")
        ]
        if not dialogue_files:
            raise FileNotFoundError(
                f"Could not find any dialogue files in {dialogues_dir} and its subdirectories."
            )
    dialogues = []
    for dialogue_file in dialogue_files:
        with open(dialogue_file, "r", encoding="utf-8") as f:
            dialogue_dict = yaml.safe_load(f)
        # print(f"Loading dialogue from {dialogue_file}...")
        dialogue = Dialogue(**dialogue_dict, path=dialogue_file)
        dialogues.append(dialogue)
    return dialogues_dir, dialogues


def load_user_personas(chatbot: Chatbot) -> dict[str, Persona]:
    user_personas = {}
    user_personas_dir = chatbot.base_directory / "user_personas"
    if not user_personas_dir.exists():
        return {}
    for file in os.listdir(user_personas_dir):
        if not file.endswith(".yaml"):
            continue
        with open(user_personas_dir / file, "r", encoding="utf-8") as f:
            user_persona_dict = yaml.safe_load(f)
            user_persona = Persona(**user_persona_dict)
            user_personas[user_persona.persona_id] = user_persona
    return user_personas
