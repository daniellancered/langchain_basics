import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant that can provide weather information for a given location. "
    "You always crack jokes and humorous but still being helpful. Respond in a short sentence"
)


@tool(
    "get_weather",
    description="Get the current weather for a given location.",
    return_direct=False,
)
def get_weather(location: str):
    res = requests.get(f"http://wttr.in/{location}?format=j1", timeout=10)
    return res.json()


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
)

conversation = [
    {"role": "user", "content": "What's the weather like in San Pedro Laguna?"}
]

response = agent.invoke({"messages": conversation})

print("Agent response:", response)
ai_response = response["messages"][-1].content

for message in ai_response:
    if message["type"] == "text":
        print(message["text"])
