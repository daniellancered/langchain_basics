from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

model = init_chat_model(
    model="auto",
    model_provider="openai",
)

# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Describe the image"},
#             {"type": "image", "url": "https://placehold.co/600x400/orange/white"},
#         ],
#     },
# ]

conversation = [
    HumanMessage(
        content=[
            {"type": "text", "text": "Describe the image"},
            {"type": "image", "url": "https://placehold.co/600x400/orange/white"},
        ],
    ),
]

response = model.invoke(conversation)
print(response.text)
