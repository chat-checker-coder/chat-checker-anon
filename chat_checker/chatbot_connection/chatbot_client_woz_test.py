from chat_checker.chatbot_connection.chatbot_client_base import ChatbotClientInterface


def chat_with_chatbot(chatbot_client: ChatbotClientInterface):
    chatbot_client.set_up_class()

    first_chatbot_message = chatbot_client.set_up_chat()
    print(f"First chatbot message: {first_chatbot_message}")

    # Get chatbot response
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit"):
                break
            response, ended = chatbot_client.get_response(user_input)
            print(f"\nChatbot: {response}\n")
            if ended:
                break
    except KeyboardInterrupt:
        print("\nChat interrupted by user.")
        pass

    print(f"Tear down response: {chatbot_client.tear_down_chat()}")

    chatbot_client.tear_down_class()
