import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
import yaml

from chat_checker.models.chatbot import Chatbot


CHAT_CHECKER_BASE_DIR = Path(__file__).parent.parent


def load_chatbot(chatbot_dir: Path) -> Chatbot:
    config_path = chatbot_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Chatbot configuration file not found at {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    try:
        chatbot = Chatbot(base_directory=chatbot_dir, **config)
    except ValidationError as e:
        print(f"Failed to load chatbot from directory {chatbot_dir}: {e}")
        raise e
    return chatbot


def load_registry() -> dict[str, Chatbot]:
    registry_path = CHAT_CHECKER_BASE_DIR / "config/chatbots_registry.yaml"
    if not registry_path.exists():
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        _registry: dict[str, str] = yaml.safe_load(f)
    if not _registry:
        return {}
    chatbot_registry: dict[str, Chatbot] = {}
    for chatbot_id, chatbot_dir in _registry.items():
        chatbot = load_chatbot(Path(chatbot_dir))
        chatbot_registry[chatbot_id] = chatbot

    return chatbot_registry


def save_registry(registry: dict[str, Chatbot]):
    registry_path = CHAT_CHECKER_BASE_DIR / "config/chatbots_registry.yaml"
    os.makedirs(registry_path.parent, exist_ok=True)
    with open(registry_path, "w+", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                chatbot.id: str(chatbot.base_directory.absolute())
                for chatbot in registry.values()
            },
            f,
        )


_REGISTRY: dict[str, Chatbot] = load_registry()


def register_chatbots(chatbots_base_dir: Path, chatbot_id: Optional[str] = None):
    chatbot_folders = []
    if chatbot_id:
        print(f"Registering chatbot {chatbot_id} from directory {chatbots_base_dir}...")
        chatbot_folders = [chatbots_base_dir / chatbot_id]
    else:
        print(f"Registering chatbots from directory {chatbots_base_dir}...")
        chatbot_folders = [d for d in chatbots_base_dir.iterdir() if d.is_dir()]
    for chatbot_folder in chatbot_folders:
        if chatbot_folder.name == "__pycache__":
            continue
        print(f"Registering chatbot from directory {chatbot_folder.name}...")
        try:
            chatbot = load_chatbot(chatbot_folder)
        except FileNotFoundError as e:
            print(
                f"Failed to register chatbot from directory {chatbot_folder.name}: {e}"
            )
            continue
        if chatbot.id in _REGISTRY:
            print(f"Chatbot {chatbot.id} is already registered")
            continue
        _REGISTRY[chatbot.id] = chatbot
        print(f"Registered chatbot {chatbot.id}")
    save_registry(_REGISTRY)


def get_chatbot(chatbot_id: str) -> Chatbot:
    chatbot = _REGISTRY.get(chatbot_id)
    if not chatbot:
        raise ValueError(
            f"Chatbot {chatbot_id} not found in the registry. Please make sure to register it first."
        )
    return chatbot
