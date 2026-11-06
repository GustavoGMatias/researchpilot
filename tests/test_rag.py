from app.services.rag import DocumentIndex, chunk_text, cosine
from tests.conftest import FakeLLM


def test_cosine():
    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([0, 0], [1, 1]) == 0


def test_chunk_text():
    assert chunk_text("") == []
    chunks = chunk_text("a" * 2000, size=800, overlap=100)
    assert len(chunks) == 3 and all(len(c) <= 800 for c in chunks)


def make_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "aurora.md").write_text("O projeto Aurora teve orçamento de R$ 480.")
    (docs / "outro.txt").write_text("Nada relacionado aqui sobre jardinagem.")
    return DocumentIndex(FakeLLM(), str(docs), str(tmp_path / "index.json")), docs


def test_busca_retorna_o_mais_relevante(tmp_path):
    idx, _ = make_index(tmp_path)
    r = idx.search("orçamento do aurora")
    assert r["results"][0]["source"] == "aurora.md"
    assert r["sources"] == ["docs/aurora.md"]


def test_sem_resultado_relevante(tmp_path):
    idx, _ = make_index(tmp_path)
    r = idx.search("zzz")
    assert r["results"] == [] and "note" in r


def test_indexacao_incremental(tmp_path):
    idx, docs = make_index(tmp_path)
    assert idx.sync() == 2
    assert idx.sync() == 0
    (docs / "aurora.md").write_text("texto novo sobre aurora")
    assert idx.sync() == 1
    (docs / "outro.txt").unlink()
    idx.sync()
    assert {e["source"] for e in idx.entries} == {"aurora.md"}


def test_indice_persiste(tmp_path):
    idx, docs = make_index(tmp_path)
    idx.sync()
    again = DocumentIndex(FakeLLM(), str(docs), str(tmp_path / "index.json"))
    assert again.sync() == 0


def test_pasta_vazia(tmp_path):
    idx = DocumentIndex(FakeLLM(), str(tmp_path / "vazia"), str(tmp_path / "i.json"))
    assert idx.search("x")["results"] == []
