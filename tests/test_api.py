import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AgentResult
from app.api.main import create_app
from app.services.llm import LLMError, LLMRateLimitError, LLMTimeoutError
from app.services.rag import DocumentIndex
from tests.conftest import FakeLLM


class FakeAgent:
    def __init__(self, error=None):
        self.error = error

    def chat(self, session_id, message):
        if self.error:
            raise self.error
        return AgentResult(f"eco: {message}", ["u"], ["search_wikipedia"], 2)


@pytest.fixture
def index(tmp_path):
    return DocumentIndex(FakeLLM(), str(tmp_path / "docs"), str(tmp_path / "i.json"))


def client(agent, index=None):
    return TestClient(create_app(agent, index))


def test_health(index):
    with client(FakeAgent(), index) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_chat(index):
    with client(FakeAgent(), index) as c:
        r = c.post("/chat", json={"session_id": "s1", "message": "oi"})
    assert r.status_code == 200
    assert r.json() == {
        "session_id": "s1", "answer": "eco: oi", "sources": ["u"],
        "tools_used": ["search_wikipedia"], "iterations": 2,
    }


@pytest.mark.parametrize("body", [
    {"session_id": "s1", "message": ""},
    {"session_id": "../x", "message": "oi"},
    {"message": "oi"},
])
def test_chat_validacao(index, body):
    with client(FakeAgent(), index) as c:
        assert c.post("/chat", json=body).status_code == 422


@pytest.mark.parametrize("err,status", [
    (LLMRateLimitError("x"), 429), (LLMTimeoutError("x"), 504), (LLMError("x"), 502),
])
def test_erros_do_llm(index, err, status):
    with client(FakeAgent(err), index) as c:
        r = c.post("/chat", json={"session_id": "s", "message": "oi"})
    assert r.status_code == status and "detail" in r.json()


def test_upload_documento(index, tmp_path):
    with client(FakeAgent(), index) as c:
        r = c.post("/documents", files={"file": ("nota.md", "projeto aurora".encode(), "text/markdown")})
        assert r.status_code == 201 and r.json() == {"filename": "nota.md", "chunks_indexed": 1}
        assert c.get("/health").json()["indexed_chunks"] == 1
    assert (tmp_path / "docs" / "nota.md").read_text() == "projeto aurora"


def test_upload_nome_malicioso_fica_em_docs(index, tmp_path):
    with client(FakeAgent(), index) as c:
        r = c.post("/documents", files={"file": ("../../evil.md", b"x", "text/markdown")})
    assert r.json()["filename"] == "evil.md"
    assert (tmp_path / "docs" / "evil.md").exists()


@pytest.mark.parametrize("name,data,status", [
    ("a.pdf", b"x", 415), ("a.txt", b"  ", 422), ("a.txt", b"\xff\xfe", 422),
    ("a.txt", b"x" * 1_000_001, 413),
])
def test_upload_rejeitado(index, name, data, status):
    with client(FakeAgent(), index) as c:
        assert c.post("/documents", files={"file": (name, data)}).status_code == status
