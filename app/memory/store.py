class SessionMemory:
    """Histórico por sessão: lista de {"role": "user"|"model", "text": str}."""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}

    def get(self, session_id: str) -> list[dict]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, text: str) -> None:
        self._sessions.setdefault(session_id, []).append({"role": role, "text": text})
