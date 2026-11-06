import json
import logging
import os
import tempfile
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class SessionMemory:
    """Histórico por sessão {"role": "user"|"model", "text": str}, persistido em JSON."""

    def __init__(self, path: str | None = None, max_messages: int = 40):
        self.path = Path(path) if path else None
        self.max_messages = max_messages
        self._lock = threading.Lock()
        self._sessions: dict[str, list[dict]] = {}
        if self.path and self.path.exists():
            try:
                self._sessions = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.exception("memória corrompida em %s; começando vazia", self.path)

    def get(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, text: str) -> None:
        with self._lock:
            msgs = self._sessions.setdefault(session_id, [])
            msgs.append({"role": role, "text": text})
            del msgs[: -self.max_messages]
            self._save()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
