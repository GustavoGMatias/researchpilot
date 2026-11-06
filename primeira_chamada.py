import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
chave = os.getenv("API_KEY")

client = genai.Client(
    api_key=chave,
    http_options=types.HttpOptions(timeout=30000)
)

resposta = client.models.generate_content(
    model="models/gemini-3.5-flash-lite",
    contents="Explique o que é um agente de IA em duas frases.",
    config=types.GenerateContentConfig(
        temperature=0,
        system_instruction="Você é um professor que explica com analogias do dia a dia."
    )
)

print(resposta.text)
