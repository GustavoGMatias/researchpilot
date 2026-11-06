import sys

from app import config
from app.agent.agent import Agent
from app.agent.tools import ToolRegistry
from app.memory.store import SessionMemory
from app.services.llm import GeminiLLM, LLMError
from app.services.rag import DocumentIndex


def build_agent() -> Agent:
    llm = GeminiLLM()
    index = DocumentIndex(llm, config.DOCS_DIR, f"{config.DATA_DIR}/index.json")
    memory = SessionMemory(f"{config.DATA_DIR}/sessions.json")
    return Agent(llm, memory, ToolRegistry(index))


def main():
    agent = build_agent()
    session = sys.argv[1] if len(sys.argv) > 1 else "cli"
    print(f"ResearchPilot (sessão '{session}') — digite 'sair' para encerrar.")
    while True:
        try:
            msg = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("sair", "exit"):
            break
        if not msg:
            continue
        try:
            r = agent.chat(session, msg)
        except LLMError as e:
            print(f"\n[erro] {e}", file=sys.stderr)
            continue
        print("\npilot>", r.answer)
        if r.tools_used:
            print(f"[ferramentas: {', '.join(r.tools_used)} | passos: {r.iterations}]")


if __name__ == "__main__":
    main()
