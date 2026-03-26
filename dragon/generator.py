from dotenv import load_dotenv
import requests
import os
import json


load_dotenv()
OPENROUTER_TOKEN = os.getenv('OPENROUTER_TOKEN')


class LLMGenerator:

    token: str = "token"
    model: str = "arcee-ai/trinity-large-preview:free"

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
                    Очень краткий ответ:
                """
        
        answer = self.__invoke(prompt)
        return answer

