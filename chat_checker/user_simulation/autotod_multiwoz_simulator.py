from typing import List, Optional

import requests

from chat_checker.models.dialogue import DialogueTurn
from chat_checker.user_simulation.user_simulator_base import (
    UserSimulatorBase,
    UserSimulatorResponse,
)
from chat_checker.user_simulation.simulator_woz_test import (
    chat_with_user_simulator,
)


class AutotodMultiwozSimulator(UserSimulatorBase):
    base_url = "http://127.0.0.1:8083"

    def __init__(
        self,
        multiwoz_dialogue_id: str,
        model: str = "gpt-3.5-turbo-1106",  # closest to gpt-3.5-turbo-0613 model in paper
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(model, temperature, seed)
        self.mwoz_dialogue_id = multiwoz_dialogue_id
        self.session = None

    def set_up_session(self, **kwargs):
        self.session = requests.Session()
        payload = {
            "dialogue_id": self.mwoz_dialogue_id,
            "model_name": self.model,
        }
        response = self.session.post(f"{self.base_url}/init-session", json=payload)
        response.raise_for_status()
        data = response.json()
        self.mwoz_dialogue_id = data.get("dialogue_id", self.mwoz_dialogue_id)

    def tear_down_session(self):
        self.mwoz_dialogue_id = None
        self.session = None

    def generate_response(
        self, chat_history: List[DialogueTurn]
    ) -> UserSimulatorResponse:
        if not self.session:
            raise ValueError("Session not set up.")
        if chat_history:
            chatbot_message = chat_history[-1].content
            payload = {"chatbot_message": chatbot_message}
        else:
            payload = {"chatbot_message": ""}
        response = self.session.post(f"{self.base_url}/get-answer", json=payload)
        response.raise_for_status()
        chatbot_response: dict = response.json()
        return UserSimulatorResponse(
            response_message=chatbot_response.get("user_answer", ""),
            is_end=chatbot_response.get("is_end", False),
            prompt_messages=[],
            model_response=None,
        )


if __name__ == "__main__":
    user_simulator = AutotodMultiwozSimulator(multiwoz_dialogue_id="SNG01856.json")
    setup_kwargs: dict = {}
    chat_with_user_simulator(user_simulator, setup_kwargs, user_initiates=True)
