from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    location: str
    celcius: float
    farenheit: float
    humidity: float


SYSTEM_PROMPT = """
You are a weather assistant.

When the user asks for weather but does not specify a location,
you MUST first call the get_user tool to determine their default location.

After obtaining the location,
call get_weather using that location.

Never ask the user for their location if get_user can provide it.
"""


@tool(
    "get_weather",
    description="Get the current weather for a given location.",
    return_direct=False,
)
def get_weather(location: str):
    res = requests.get(f"http://wttr.in/{location}?format=j1", timeout=10)
    return res.json()


@tool(
    "get_user",
    description="Get user's location based on context",
    return_direct=False,
)
def get_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "123":
            return "Laguna"
        case "456":
            return "San Pedro"
        case _:
            return "Unknown"


llm = init_chat_model(
    model="auto",
    model_provider="openai",
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather, get_user],
    system_prompt=SYSTEM_PROMPT,
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

conversation = [{"role": "user", "content": "What's the weather like?"}]

config = {"configurable": {"thread_id": 1}}

response = agent.invoke({"messages": conversation},
    config=config, 
    context=Context(user_id="123")
)

# print("Agent response:", response)
# ai_response = response["messages"][-1].content  

# for message in ai_response:
#     if message["type"] == "text":
#         print(message["text"])

# structured response
print(response["structured_response"])
