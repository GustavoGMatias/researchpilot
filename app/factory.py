from app import config
from app.agent.agent import Agent
from app.agent.tools import ToolRegistry
from app.memory.store import SessionMemory
from app.services.llm import GeminiLLM
from app.services.rag import DocumentIndex


def build() -> tuple[Agent, DocumentIndex]:
    llm = GeminiLLM()
    index = DocumentIndex(llm, config.DOCS_DIR, f"{config.DATA_DIR}/index.json")
    memory = SessionMemory(f"{config.DATA_DIR}/sessions.json")
    return Agent(llm, memory, ToolRegistry(index)), index
