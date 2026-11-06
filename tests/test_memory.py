from app.memory.store import SessionMemory


def test_persistencia(tmp_path):
    p = tmp_path / "s.json"
    SessionMemory(str(p)).append("a", "user", "oi")
    assert SessionMemory(str(p)).get("a") == [{"role": "user", "text": "oi"}]


def test_limite_de_mensagens(tmp_path):
    m = SessionMemory(str(tmp_path / "s.json"), max_messages=3)
    for i in range(6):
        m.append("a", "user", str(i))
    assert [x["text"] for x in m.get("a")] == ["3", "4", "5"]


def test_arquivo_corrompido(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{quebrado")
    assert SessionMemory(str(p)).get("a") == []


def test_clear(tmp_path):
    m = SessionMemory(str(tmp_path / "s.json"))
    m.append("a", "user", "x")
    m.clear("a")
    assert m.get("a") == []
