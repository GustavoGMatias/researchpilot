import sys

from app.agent.agent import Agent
from app.memory.store import SessionMemory
from app.services.llm import GeminiLLM, LLMError


def main():
    agent = Agent(GeminiLLM(), SessionMemory())
    print("ResearchPilot — digite 'sair' para encerrar.")
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
            print("\npilot>", agent.chat("cli", msg))
        except LLMError as e:
            print(f"\n[erro] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
