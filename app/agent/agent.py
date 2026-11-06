import logging

from google.genai import types

from app.memory.store import SessionMemory

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "Você é o ResearchPilot, um assistente de pesquisa. Responda em português, de forma clara e objetiva."


def _to_contents(history: list[dict]) -> list[types.Content]:
    return [
        types.Content(role=m["role"], parts=[types.Part.from_text(text=m["text"])])
        for m in history
    ]


class Agent:
    def __init__(self, llm, memory: SessionMemory):
        self.llm = llm
        self.memory = memory

    def chat(self, session_id: str, message: str) -> str:
        history = self.memory.get(session_id)
        history.append({"role": "user", "text": message})
        response = self.llm.generate(_to_contents(history), SYSTEM_PROMPT)
        answer = response.text or ""
        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "model", answer)
        return answer
