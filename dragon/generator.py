from dotenv import load_dotenv
import requests
import os
import asyncio
import json
from gigachat import GigaChat

load_dotenv()
OPENROUTER_TOKEN = os.getenv('OPENROUTER_TOKEN3')
GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")


class GigaChatGenerator:

    token: str = "token"
    # model: str = "arcee-ai/trinity-large-preview:free"
    model: str = "Gigachat"
    

    def __init__(self):
        self.token = GIGACHAT_TOKEN
    
    def __invoke(self, prompt):
        with GigaChat(credentials=self.token) as giga:
            response = giga.chat(prompt)
        
        return response.choices[0].message.content
    
    def generate(self, context, query):
        
        prompt = f"""Ты - ассистент, который отвечает на вопросы строго на основе предоставленного контекста. используешься для RAG системы. Не нужны пояснения.
                    Давай ответ или предположение в кратком виде (Пример: Кто президент РФ в 2022? Ответ: Владимир Владимирович Путин).
                    Контекст: {context}
                    Вопрос\Запрос: {query}
                """
        
        answer = self.__invoke(prompt)
        return answer


class OpenRouterGenerator:

    token: str = "token"
    # model: str = "arcee-ai/trinity-large-preview:free"
    model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    

    def __init__(self):
        self.token = OPENROUTER_TOKEN
    
    def __invoke(self, prompt):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f'Bearer {OPENROUTER_TOKEN}',
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                }),
                timeout=30
            )
        except requests.exceptions.RequestException as e:
            print(f"Сетевая ошибка: {e}")
            return "Ошибка соединения с API"

        if response.status_code != 200:
            print(f"Ошибка API. Код: {response.status_code}")
            print("Ответ:", response.text)
            return f"Ошибка API: {response.status_code}"

        data = response.json()
        if 'error' in data:
            print("Ошибка в ответе OpenRouter:", data['error'])
            return f"Ошибка модели: {data['error'].get('message', 'неизвестная ошибка')}"

        if 'choices' not in data or len(data['choices']) == 0:
            print("Неожиданный формат ответа:", data)
            return "Пустой ответ от модели"

        return data['choices'][0]['message']['content']
    
    def generate(self, context, query):
        
        prompt = f"""Ты - ассистент, который отвечает на вопросы строго на основе предоставленного контекста.
                    Контекст: {context}
                    Вопрос: {query}
                """
        
        answer = self.__invoke(prompt)
        return answer

