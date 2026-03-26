from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
OPENROUTER_TOKEN = os.getenv('OPENROUTER_TOKEN')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_TOKEN
)


response = client.chat.completions.create(
    model="arcee-ai/trinity-large-preview:free",  
    messages=[
        {"role": "user", "content": "Привет, как дела?"}
    ],
)

print(response.choices[0].message.content)