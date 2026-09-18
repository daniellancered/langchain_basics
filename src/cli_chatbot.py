from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant that can provide weather information for a given location. "
    "You always crack jokes and humorous but still being helpful. Respond in a short sentence"
)

model = init_chat_model(
    model="auto",
    model_provider="openai",
)

conversation = [
    SystemMessage(SYSTEM_PROMPT),
]

while True:
    user_input = input("You: ")

    if user_input == 'quit':
        print("Assistant: Thank you! Goodbye!")
        break

    conversation.append(HumanMessage(user_input))

    ai_response = ""
    print("Assistant: ", end="", flush=True)
    for chunk in model.stream(conversation):
        if isinstance(chunk, AIMessageChunk):
            print(chunk.content, end="", flush=True)
            ai_response += chunk.content

    print("")
    conversation.append(AIMessage(ai_response))
