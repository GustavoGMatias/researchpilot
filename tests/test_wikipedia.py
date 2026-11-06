import httpx
import pytest

from app.services import wikipedia


class Resp:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=None)

    def json(self):
        return self.data


def test_parse(monkeypatch):
    data = {"query": {"pages": {
        "2": {"title": "B", "fullurl": "u2", "extract": "tb", "index": 2},
        "1": {"title": "A", "fullurl": "u1", "extract": "ta", "index": 1},
    }}}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(data))
    r = wikipedia.search_wikipedia("x")
    assert [x["title"] for x in r["results"]] == ["A", "B"]
    assert r["sources"] == ["u1", "u2"]


def test_sem_resultados(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp({"batchcomplete": ""}))
    assert wikipedia.search_wikipedia("x")["results"] == []


def test_erro_http(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp({}, 500))
    with pytest.raises(wikipedia.WikipediaError):
        wikipedia.search_wikipedia("x")


def test_timeout(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectTimeout("t")
    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(wikipedia.WikipediaError):
        wikipedia.search_wikipedia("x")
