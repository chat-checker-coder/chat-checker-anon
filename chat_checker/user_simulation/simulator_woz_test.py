from typing import Dict, Any, List
from chat_checker.models.dialogue import DialogueTurn, SpeakerRole
from chat_checker.user_simulation.user_simulator_base import UserSimulatorBase


def chat_with_user_simulator(
    user_simulator: UserSimulatorBase,
    setup_kwargs: Dict[str, Any],
    user_initiates: bool = False,
) -> None:
    user_simulator.set_up_session(**setup_kwargs)

    chat_history: List[DialogueTurn] = []
    print("You play the role of the chatbot.")
    if user_initiates:
        simulator_response = user_simulator.generate_response(chat_history)
        turn = DialogueTurn(
            turn_id=1,
            role=SpeakerRole.DIALOGUE_SYSTEM,
            content=simulator_response.response_message,
        )
        chat_history.append(turn)
        print(f"User: {simulator_response.response_message}")

    try:
        while True:
            user_input = input("You (chatbot): ")
            if user_input.lower() in ("exit", "quit"):
                break
            chat_history.append(
                DialogueTurn(
                    turn_id=len(chat_history) + 1,
                    role=SpeakerRole.USER,
                    content=user_input,
                )
            )
            simulator_response = user_simulator.generate_response(chat_history)
            print(f"User: {simulator_response.response_message}\n")
            if simulator_response.is_end:
                break
    except KeyboardInterrupt:
        print("\nChat interrupted.")
        pass

    user_simulator.tear_down_session()
