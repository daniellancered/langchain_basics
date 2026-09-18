from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

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
    HumanMessage("What's the weather like in San Pedro Laguna?"),
]

for chunk in model.stream(conversation):
    print(chunk.text, end="", flush=True)
