import sys

from app import factory
from app.services.llm import LLMError


def main():
    agent, _ = factory.build()
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
