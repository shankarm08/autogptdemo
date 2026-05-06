import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)

# Create kernel
kernel = Kernel()

# Connect Ollama
service = OllamaChatCompletion(
    ai_model_id="llama3.2:1b",
    host="http://localhost:11434"
)

kernel.add_service(service)

# Settings
settings = PromptExecutionSettings()

# Chat history
chat_history = ChatHistory()

print("🤖 AI Chatbot Started")
print("Type 'exit' to stop\n")


async def chat():

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Chatbot stopped.")
            break

        # Add user message
        chat_history.add_user_message(user_input)

        # Get AI response
        response = await service.get_chat_message_content(
            chat_history=chat_history,
            settings=settings
        )

        # Print response
        print("AI:", response)

        # Save AI response
        chat_history.add_assistant_message(str(response))


# Run chatbot
asyncio.run(chat())