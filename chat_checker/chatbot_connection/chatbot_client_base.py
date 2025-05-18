from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any


class ChatbotClientInterface(ABC):
    """
    Abstract base class for interacting with different chatbots.
    """

    @classmethod
    def set_up_class(cls) -> Any:
        """
        Set up resources needed for the class.

        This method is intended to perform any initialization that is required
        before interacting with the chatbot, such as setting up API keys,
        configuring logging, or establishing any global settings.

        Returns:
            Any: Optional return value that may be used to confirm setup success.
        """
        pass

    @classmethod
    def tear_down_class(cls) -> Any:
        """
        Tear down resources allocated for the class.

        This method is intended to release any resources that were allocated
        during the class setup, such as closing connections or cleaning up
        temporary data.

        Returns:
            Any: Optional return value that may be used to confirm teardown success.
        """
        pass

    @abstractmethod
    def set_up_chat(self, *args) -> Optional[str]:
        """
        Set up resources needed for an individual chat session.

        This method is used to initialize a new chat session with the chatbot, such as
        preparing the session parameters or sending an initial request to the server.

        Args:
            *args: Variable arguments that may be required to set up the chat.

        Returns:
            Optional[str]: Greeting/conversation initiation message from the chatbot or None. None --> the user should initiate the conversation.
        """
        pass

    @abstractmethod
    def tear_down_chat(self, *args) -> Any:
        """
        Tear down resources for an individual chat session.

        This method is used to clean up after a chat session is complete, such as
        sending a termination signal or releasing session-specific resources.

        Args:
            *args: Variable arguments that may be required to tear down the chat.

        Returns:
            Any: Response or status indicating the success of the teardown.
        """
        pass

    @abstractmethod
    def get_response(self, user_message: str) -> Tuple[str, bool]:
        """
        Send a user message to the chatbot and get the response.

        Args:
            user_message (str): The message from the user to be sent to the chatbot.

        Returns:
            Tuple[str, bool]: The chatbot's response and a boolean indicating if the conversation has ended.
        """
        pass
