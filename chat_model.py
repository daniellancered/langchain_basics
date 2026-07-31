import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant that can provide weather information for a given location. You always crack jokes and humorous but still being helpful. Respond in a short sentence"


model = init_chat_model(
    model="google_genai:gemini-3.6-flash"
)

conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather like in San Pedro Laguna?"}
]

response = model.invoke(conversation)
ai_response = response.content

for message in ai_response:
    if message['type'] == "text":
        print(message['text'])
        
        