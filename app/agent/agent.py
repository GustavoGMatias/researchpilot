import logging
from dataclasses import dataclass, field

from google.genai import types

from app import config
from app.memory.store import SessionMemory

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é o ResearchPilot, um assistente de pesquisa. Responda em português, de forma clara e objetiva.
Ferramentas: search_wikipedia (conhecimento geral) e search_documents (documentos locais do usuário).
Regras:
- Para fatos, use as ferramentas antes de responder; não invente.
- Baseie a resposta apenas no que as ferramentas retornaram e cite no fim, em "Fontes:", somente as fontes (URL ou docs/arquivo) que realmente sustentam a resposta.
- Se as ferramentas não trouxerem informação suficiente, diga isso claramente, não use conhecimento próprio como substituto e não liste "Fontes:".
- Se uma ferramenta retornar erro, avise o usuário e use a outra se fizer sentido.
- Conversa simples (saudações, perguntas sobre a própria conversa) não precisa de ferramentas nem de fontes."""

LIMIT_MESSAGE = "Não consegui concluir a pesquisa dentro do limite de passos. Tente reformular a pergunta."


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0


def _to_contents(history: list[dict]) -> list[types.Content]:
    return [
        types.Content(role=m["role"], parts=[types.Part.from_text(text=m["text"])])
        for m in history
    ]


class Agent:
    def __init__(self, llm, memory: SessionMemory, tools, max_iterations: int = config.MAX_ITERATIONS):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.max_iterations = max_iterations

    def chat(self, session_id: str, message: str) -> AgentResult:
        history = self.memory.get(session_id)
        history.append({"role": "user", "text": message})
        contents = _to_contents(history)
        sources: list[str] = []
        used: list[str] = []
        answer, iterations = LIMIT_MESSAGE, 0

        for iterations in range(1, self.max_iterations + 1):
            response = self.llm.generate(contents, SYSTEM_PROMPT, self.tools.tools)
            calls = response.function_calls or []
            if not calls:
                answer = response.text or ""
                break
            contents.append(response.candidates[0].content)
            parts = []
            for call in calls:
                log.info("tool call: %s %s", call.name, dict(call.args or {}))
                result = self.tools.run(call.name, dict(call.args or {}))
                used.append(call.name)
                for s in result.get("sources", []):
                    if s not in sources:
                        sources.append(s)
                parts.append(types.Part.from_function_response(name=call.name, response=result))
            contents.append(types.Content(role="user", parts=parts))
        else:
            log.warning("limite de %d iterações atingido", self.max_iterations)

        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "model", answer)
        return AgentResult(answer, sources, used, iterations)
